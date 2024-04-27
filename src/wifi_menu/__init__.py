from __future__ import annotations

import argparse as ap
import asyncio
import logging as _logging
import operator as op
import os
import subprocess as sp
import shlex
import sys
import textwrap
from contextlib import suppress
from enum import IntEnum, unique
from functools import cache
from itertools import starmap, tee
from typing import TYPE_CHECKING, cast

import sdbus
from fzf_but_typed import fzf_iter, SearchOptions, LayoutOptions, LayoutType, InterfaceOptions, Binding, Key, ActionSimple, ActionWithArg, ActionWithArgType, ExitStatusCode as FzfExitStatus
from sdbus_async.networkmanager import (NetworkManager, DeviceType, NetworkDeviceGeneric,
                                        NetworkDeviceWireless, NetworkManagerSettings,
                                        NetworkConnectionSettings, AccessPoint, ActiveConnection,
                                        ConnectionState)
from sdbus_async.networkmanager.exceptions import NmAgentManagerNoSecretsError
from sdbus_async.networkmanager.settings import \
        ConnectionProfile, ConnectionSettings, WirelessSettings, WirelessSecuritySettings

try:
    from sdbus_async.notifications import FreedesktopNotifications
except ModuleNotFoundError:
    pass

_logging.basicConfig(level=_logging.WARNING)
logger = _logging.getLogger(__name__)


def _is_sorted(seq: Sequence[_T], cmp_fn: Callable[[_T, _T], bool] | None = None) -> bool:
    fn = cast(Callable[[_T, _T], bool], op.le) if cmp_fn is None else cmp_fn
    seq1, seq2 = tee(seq)
    next(seq2, None)
    return all(starmap(fn, zip(seq1, seq2)))


_CLI_LOGGING_LEVELS = [
    _logging.WARNING,
    _logging.INFO,
    _logging.DEBUG,
]
assert _is_sorted(_CLI_LOGGING_LEVELS, cmp_fn=op.ge)

if TYPE_CHECKING:
    from collections.abc import Iterable, Awaitable, Sequence, Callable
    from typing import Protocol, TypeVar, Any
    _T = TypeVar("_T")

    class SupportsStr(Protocol):

        def __str__(self) -> str:
            ...


APP_NAME = os.path.basename(sys.argv[0])
APP_NAME_QUOTED = shlex.quote(APP_NAME)
APP_DISPLAY_NAME = "Wifi Menu"
APP_DISPLAY_NAME_QUOTED = shlex.quote(APP_DISPLAY_NAME)
FZF_HEADER_PARTS: list[str] = []

# yapf: disable
FZF_HEADER_PARTS.append("ctrl-d: delete profile")
FZF_CMD_DELETE_PROFILE = textwrap.dedent("""\
    if ! nmcli --fields="" connection show id {{}}; then
        notify-send \\
            --app-name={app_name} \\
            --icon=network-wireless-disconnected \\
            {app_display_name} \\
            "No profile found for {{}}"
            exit 0
    fi
    if nmcli connection delete {{}} >&2; then
        notify-send \\
            --app-name={app_name} \\
            --icon=network-wireless-disconnected \\
            {app_display_name} \\
            "Deleted profile for {{}}"
        exit 0
    fi
    notify-send \\
        --app-name={app_name} \\
        --icon=network-wireless-disconnected \\
        {app_display_name} \\
        "Couldn't delete profile for {{}}"
    exit 1
""")

FZF_HEADER_PARTS.append("ctrl-r: rescan wifi access points")
FZF_CMD_RESCAN = textwrap.dedent("""\
    nmcli --terse --fields=SSID device wifi list --rescan yes
""")

# yapf: enable


@unique
class ExitStatus(IntEnum):
    SUCCESS = 0
    NOT_CHANGED = 1
    GENERIC_ERROR = 2
    NO_WIFI_DEVICE = 3
    NO_ACCESS_POINT = 4


@cache
def get_nm() -> NetworkManager:
    sdbus.set_default_bus(sdbus.sd_bus_open_system())
    return NetworkManager()


@cache
def get_nm_settings() -> NetworkManagerSettings:
    return NetworkManagerSettings()


@cache
async def get_nm_all_connection_settings() -> dict[str, NetworkConnectionSettings]:
    connections: list[str] = await get_nm_settings().connections
    return {path: NetworkConnectionSettings(path) for path in connections}


@cache
async def get_nm_devices() -> dict[str, NetworkDeviceGeneric]:
    devices: list[str] = await get_nm().devices
    return {path: NetworkDeviceGeneric(path) for path in devices}


async def _is_wifi_dev(path: str, dev: NetworkDeviceGeneric) -> str | None:
    if (await dev.device_type) == DeviceType.WIFI:
        return path
    return None


async def _filter_wifi_devs(
    devs: Iterable[tuple[str, NetworkDeviceGeneric]],
) -> Iterable[tuple[str, NetworkDeviceWireless]]:
    paths = filter(None, await asyncio.gather(*starmap(_is_wifi_dev, devs)))
    return ((path, NetworkDeviceWireless(path)) for path in paths)


@cache
async def get_nm_wifi_devices() -> dict[str, NetworkDeviceWireless]:
    devices = await get_nm_devices()
    return dict(await _filter_wifi_devs(devices.items()))


async def _ignore_no_secrets(coro: Awaitable[_T]) -> _T | None:
    with suppress(NmAgentManagerNoSecretsError):
        return await coro
    return None


async def _get_cs_profiles(
    ncs_many: Iterable[NetworkConnectionSettings],
) -> Iterable[ConnectionProfile | None]:
    return await asyncio.gather(*(_ignore_no_secrets(ncs.get_profile()) for ncs in ncs_many))


async def get_access_points(dev: NetworkDeviceWireless) -> dict[str, AccessPoint]:
    access_points: list[str] = await dev.access_points
    return {path: AccessPoint(path) for path in access_points}


async def _get_ssid_ap_pair(ap: AccessPoint) -> tuple[str, AccessPoint]:
    return cast(bytes, await ap.ssid).decode('utf-8', errors="replace"), ap


def get_args() -> dict[str, Any]:
    parser = ap.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="count", default=0, help="Can be used up to 3 times")
    parser.add_argument('-s', '--scan', action="store_true", default=False)
    return vars(parser.parse_args())


def fzf_wrapper(data: Iterable[SupportsStr], **kwargs) -> list[str]:
    try:
        if (so := kwargs.get('search', None)) is not None:
            so.exact = True
        else:
            kwargs['search'] = SearchOptions(exact=False)
        return fzf_iter(data, **kwargs)
    except sp.CalledProcessError as error:
        match error.returncode:
            case FzfExitStatus.NO_MATCH:
                logger.info("No such item to select")
                raise SystemExit(ExitStatus.NOT_CHANGED)
            case FzfExitStatus.USER_INTERRUPTED:
                logger.info("Selection aborted by user.")
                raise SystemExit(ExitStatus.NOT_CHANGED)
            case FzfExitStatus.ERROR:
                logger.error("'fzf' returned an error: %s", '\n'.join(error.stderr))
                raise SystemExit(ExitStatus.GENERIC_ERROR)
            case other:
                raise NotImplementedError()


def new_profile(ssid: str, interface_id: str) -> ConnectionProfile:
    from getpass import getpass
    return ConnectionProfile(
        connection=ConnectionSettings(
            autoconnect=True,
            connection_id=ssid,
            connection_type="802-11-wireless",
            interface_name=interface_id,
        ),
        wireless=WirelessSettings(ssid=ssid.encode("utf-8")),
        wireless_security=WirelessSecuritySettings(key_mgmt="wpa-psk", psk=getpass()),
    )


async def main(verbose: int = 0, scan: bool = False) -> int:
    if verbose >= len(_CLI_LOGGING_LEVELS):
        verbose = len(_CLI_LOGGING_LEVELS) - 1
    logger.setLevel(_CLI_LOGGING_LEVELS[verbose])

    if len(wifi_devs := await get_nm_wifi_devices()) == 0:
        logger.info("Couldn't find wireless device from NetworkManager. Exiting.")
        return ExitStatus.NO_WIFI_DEVICE
    wifi_dev: NetworkDeviceWireless
    wifi_dev_path, wifi_dev = next(iter(wifi_devs.items()))
    wifi_interface: str = await wifi_dev.interface
    logger.debug("Using wireless device: %s (interface=%s)", wifi_dev_path, wifi_interface)

    if scan:
        logger.debug("Scanning for wifi networks, from %s", wifi_interface)
        await wifi_dev.request_scan({})
        async for interface_name, changed_props, invalidated_props in wifi_dev.properties_changed:
            if 'LastScan' in changed_props:
                break
        logger.debug("Finished scanning")

    if len(ap_paths := cast(list[str], await wifi_dev.access_points)) == 0:
        logger.info("Device on interface '%s' hasn't detected any access points. Exiting.", wifi_interface)
        return ExitStatus.NO_ACCESS_POINT
    access_points = [AccessPoint(path) for path in cast(list[str], ap_paths)]
    ap_by_ssid: dict[str, AccessPoint] = \
        dict(await asyncio.gather(*map(_get_ssid_ap_pair, access_points)))
    ssids = list(ap_by_ssid.keys())

    cmd_delete_profile = FZF_CMD_DELETE_PROFILE.format(
        app_name=APP_NAME_QUOTED,
        app_display_name=APP_DISPLAY_NAME_QUOTED,
    )
    logger.debug(cmd_delete_profile)
    # yapf: disable
    result = fzf_wrapper(ap_by_ssid.keys(),
        layout=LayoutOptions(
            layout=LayoutType.REVERSE_LIST,
            header="\n".join(FZF_HEADER_PARTS),
            header_first=True,
        ),
        interface=InterfaceOptions(bind=[
            Binding(binding=Key.CTRL_D, actions=[
                ActionWithArg(
                    action_type=ActionWithArgType.EXECUTE,
                    argument=cmd_delete_profile),
            ]),
            Binding(binding=Key.CTRL_R, actions=[
                ActionWithArg(
                    action_type=ActionWithArgType.RELOAD,
                    argument=FZF_CMD_RESCAN)
            ]),
            Binding(binding=Key.ESC, actions=[ActionSimple.CANCEL]),
        ])
    )
    chosen_ap_ssid = result[0]
    if chosen_ap_ssid not in ssids:
        raise RuntimeError(f"Chose a newly discovered network, of ssid: {chosen_ap_ssid!r}")
    chosen_ap_path = cast(str, ap_by_ssid[chosen_ap_ssid]._remote_object_path)
    logger.debug("Using access point: '%s' (%s)", chosen_ap_ssid, chosen_ap_path)
    # yapf: enable

    ncs_paths: list[str] = await get_nm_settings().connections
    network_conn_settings = list(map(NetworkConnectionSettings, ncs_paths))
    profiles = await _get_cs_profiles(network_conn_settings)
    is_profile_new = False

    for i, profile in enumerate(profiles):
        if profile is None:
            continue
        if profile.connection.connection_id == chosen_ap_ssid:
            logger.debug("Found existing network profile for chosen SSID.")
            chosen_conn_path = ncs_paths[i]
            break
    else:
        logger.debug("No network profile found for chosen SSID. Creating one...")
        chosen_conn_path, _ = await get_nm_settings().add_connection_profile(
            profile=new_profile(ssid=chosen_ap_ssid, interface_id=wifi_interface),
            save_to_disk=False,
        )
        is_profile_new = True

    logger.debug("Activating connection.")
    ac_path = await get_nm().activate_connection(
        connection=chosen_conn_path,
        device=wifi_dev_path,
        specific_object=chosen_ap_path,
    )
    active_connection = ActiveConnection(ac_path)

    state_changed_signal = aiter(active_connection.state_changed)
    state, _reason = await anext(state_changed_signal)
    if state != ConnectionState.ACTIVATING:
        logger.error("Got unexpected signal from dbus/nm: %s", ConnectionState(state).name)
        return ExitStatus.GENERIC_ERROR

    state, _reason = await anext(state_changed_signal)
    match state:
        case ConnectionState.ACTIVATED:
            logger.info("Successfully connected to: %s", chosen_ap_ssid)
            try:
                notification_server = FreedesktopNotifications(bus=sdbus.sd_bus_open_user())
                _notification_id = await notification_server.notify(
                    app_name=APP_NAME,
                    app_icon='network-wireless-connected',
                    summary='Wi-Fi connection established',
                    body=f'connected to {chosen_ap_ssid!r}',
                    hints=notification_server.create_hints(urgency=1),
                )
            except NameError:
                pass
            if is_profile_new:
                ncs = NetworkConnectionSettings(chosen_conn_path)
                await ncs.get_profile()   # Without this, the next line fails
                await ncs.save()
                logger.debug("Saved profile settings for new connection")
            return ExitStatus.SUCCESS
        case ConnectionState.DEACTIVATED:
            logger.info("Failed to connect to: %s", chosen_ap_ssid)
            try:
                notification_server = FreedesktopNotifications(bus=sdbus.sd_bus_open_user())
                _notification_id = await notification_server.notify(
                    app_name=APP_NAME,
                    app_icon='network-wireless-offline',
                    summary=f"Couldn't connect to {chosen_ap_ssid!r}",
                    body=f"maybe the network password changed?",
                    expire_timeout=0,
                    hints=notification_server.create_hints(urgency=2),
                )
            except NameError:
                pass
            if is_profile_new:
                ncs = NetworkConnectionSettings(chosen_conn_path)
                await ncs.get_profile()   # Without this, the next line fails
                await ncs.delete()
                logger.debug("Discarded profile settings for new connection")
            return ExitStatus.NOT_CHANGED
        case other:
            logger.error("Got unexpected signal from dbus/nm: %s", ConnectionState(state).name)
            return ExitStatus.GENERIC_ERROR


def main_wrapper() -> None:
    raise SystemExit(asyncio.run(main(**get_args())))


if __name__ == "__main__":
    main_wrapper()
