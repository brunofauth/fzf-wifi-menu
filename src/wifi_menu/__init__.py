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
import fzf_but_typed as fbt
import sdbus_async.networkmanager as nm
import sdbus_async.networkmanager.exceptions as nme
import sdbus_async.networkmanager.settings as nms

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
def get_nm() -> nm.NetworkManager:
    sdbus.set_default_bus(sdbus.sd_bus_open_system())
    return nm.NetworkManager()


@cache
def get_nm_settings() -> nm.NetworkManagerSettings:
    return nm.NetworkManagerSettings()


@cache
async def get_nm_all_connection_settings() -> dict[str, nm.NetworkConnectionSettings]:
    connections: list[str] = await get_nm_settings().connections
    return {path: nm.NetworkConnectionSettings(path) for path in connections}


@cache
async def get_nm_devices() -> dict[str, nm.NetworkDeviceGeneric]:
    devices: list[str] = await get_nm().devices
    return {path: nm.NetworkDeviceGeneric(path) for path in devices}


async def _is_wifi_dev(path: str, dev: nm.NetworkDeviceGeneric) -> str | None:
    if (await dev.device_type) == nm.DeviceType.WIFI:
        return path
    return None


async def _filter_wifi_devs(
    devs: Iterable[tuple[str, nm.NetworkDeviceGeneric]],
) -> Iterable[tuple[str, nm.NetworkDeviceWireless]]:
    paths = filter(None, await asyncio.gather(*starmap(_is_wifi_dev, devs)))
    return ((path, nm.NetworkDeviceWireless(path)) for path in paths)


@cache
async def get_nm_wifi_devices() -> dict[str, nm.NetworkDeviceWireless]:
    devices = await get_nm_devices()
    return dict(await _filter_wifi_devs(devices.items()))


async def _ignore_no_secrets(coro: Awaitable[_T]) -> _T | None:
    with suppress(nme.NmAgentManagerNoSecretsError):
        return await coro
    return None


async def _get_cs_profiles(
    ncs_many: Iterable[nm.NetworkConnectionSettings],
) -> Iterable[nms.ConnectionProfile | None]:
    return await asyncio.gather(*(_ignore_no_secrets(ncs.get_profile()) for ncs in ncs_many))


async def get_access_points(dev: nm.NetworkDeviceWireless) -> dict[str, nm.AccessPoint]:
    access_points: list[str] = await dev.access_points
    return {path: nm.AccessPoint(path) for path in access_points}


async def _get_ssid(ap: nm.AccessPoint) -> tuple[str, nm.AccessPoint]:
    return cast(bytes, await ap.ssid).decode('utf-8', errors="replace")


async def _get_ap_path_by_ssid(ap_paths: list[str]) -> dict[str, str]:
    access_points = [nm.AccessPoint(path) for path in ap_paths]
    ssids = await asyncio.gather(*map(_get_ssid, access_points))
    return dict(zip(ssids, ap_paths))


async def _get_ap_or_exit(wifi_dev: nm.NetworkDeviceWireless) -> list[str]:
    if len(ap_paths := cast(list[str], await wifi_dev.access_points)) == 0:
        logger.info("Device on interface '%s' hasn't detected any access points. Exiting.", wifi_interface)
        sys.exit(ExitStatus.NO_ACCESS_POINT)
    return ap_paths


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
            kwargs['search'] = fbt.SearchOptions(exact=False)
        return fbt.fzf_iter(data, **kwargs)
    except sp.CalledProcessError as error:
        match error.returncode:
            case fbt.ExitStatusCode.NO_MATCH:
                logger.info("No such item to select")
                raise SystemExit(ExitStatus.NOT_CHANGED)
            case fbt.ExitStatusCode.USER_INTERRUPTED:
                logger.info("Selection aborted by user.")
                raise SystemExit(ExitStatus.NOT_CHANGED)
            case fbt.ExitStatusCode.ERROR:
                logger.error("'fzf' returned an error: %s", '\n'.join(error.stderr))
                raise SystemExit(ExitStatus.GENERIC_ERROR)
            case other:
                raise NotImplementedError()


def new_profile(ssid: str, interface_id: str) -> nms.ConnectionProfile:
    from getpass import getpass
    return nms.ConnectionProfile(
        connection=nms.ConnectionSettings(
            autoconnect=True,
            connection_id=ssid,
            connection_type="802-11-wireless",
            interface_name=interface_id,
        ),
        wireless=nms.WirelessSettings(ssid=ssid.encode("utf-8")),
        wireless_security=nms.WirelessSecuritySettings(key_mgmt="wpa-psk", psk=getpass()),
    )


async def main(verbose: int = 0, scan: bool = False) -> int:
    if verbose >= len(_CLI_LOGGING_LEVELS):
        verbose = len(_CLI_LOGGING_LEVELS) - 1
    logger.setLevel(_CLI_LOGGING_LEVELS[verbose])

    if len(wifi_devs := await get_nm_wifi_devices()) == 0:
        logger.info("Couldn't find wireless device from NetworkManager. Exiting.")
        return ExitStatus.NO_WIFI_DEVICE
    wifi_dev: nm.NetworkDeviceWireless
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

    ap_paths = await _get_ap_or_exit(wifi_dev)
    ap_by_ssid = await _get_ap_path_by_ssid(ap_paths)
    ssids = list(ap_by_ssid.keys())

    cmd_delete_profile = FZF_CMD_DELETE_PROFILE.format(
        app_name=APP_NAME_QUOTED,
        app_display_name=APP_DISPLAY_NAME_QUOTED,
    )
    logger.debug(cmd_delete_profile)
    # yapf: disable
    result = fzf_wrapper(ap_by_ssid.keys(),
        layout=fbt.LayoutOptions(
            layout=fbt.LayoutType.REVERSE_LIST,
            header="\n".join(FZF_HEADER_PARTS),
            header_first=True,
        ),
        interface=fbt.InterfaceOptions(bind=[
            fbt.Binding(binding=fbt.Key.CTRL_D, actions=[
                fbt.ActionWithArg(
                    action_type=fbt.ActionWithArgType.EXECUTE,
                    argument=cmd_delete_profile),
            ]),
            fbt.Binding(binding=fbt.Key.CTRL_R, actions=[
                fbt.ActionWithArg(
                    action_type=fbt.ActionWithArgType.RELOAD,
                    argument=FZF_CMD_RESCAN)
            ]),
            fbt.Binding(binding=fbt.Key.ESC, actions=[fbt.ActionSimple.CANCEL]),
        ])
    )
    chosen_ap_ssid = result[0]
    if chosen_ap_ssid not in ssids:
        logger.info("Available networks changed since startup. Updating internal cache.")
        ap_paths = await _get_ap_or_exit(wifi_dev)
        ap_by_ssid = await _get_ap_path_by_ssid(ap_paths)
        ssids = list(ap_by_ssid.keys())
    if chosen_ap_ssid not in ssids:
        logger.critical("The network '%s' is not available anymore. Quitting.", chosen_ap_ssid)
        return ExitStatus.NOT_CHANGED
    chosen_ap_path = ap_by_ssid[chosen_ap_ssid]
    logger.debug("Using access point: '%s' (%s)", chosen_ap_ssid, chosen_ap_path)
    # yapf: enable

    ncs_paths: list[str] = await get_nm_settings().connections
    network_conn_settings = list(map(nm.NetworkConnectionSettings, ncs_paths))
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
    active_connection = nm.ActiveConnection(ac_path)

    state_changed_signal = aiter(active_connection.state_changed)
    state, _reason = await anext(state_changed_signal)
    if state != nm.ConnectionState.ACTIVATING:
        logger.error("Got unexpected signal from dbus/nm: %s", nm.ConnectionState(state).name)
        return ExitStatus.GENERIC_ERROR

    state, _reason = await anext(state_changed_signal)
    match state:
        case nm.ConnectionState.ACTIVATED:
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
                ncs = nm.NetworkConnectionSettings(chosen_conn_path)
                await ncs.get_profile()   # Without this, the next line fails
                await ncs.save()
                logger.debug("Saved profile settings for new connection")
            return ExitStatus.SUCCESS
        case nm.ConnectionState.DEACTIVATED:
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
                ncs = nm.NetworkConnectionSettings(chosen_conn_path)
                await ncs.get_profile()   # Without this, the next line fails
                await ncs.delete()
                logger.debug("Discarded profile settings for new connection")
            return ExitStatus.NOT_CHANGED
        case other:
            logger.error("Got unexpected signal from dbus/nm: %s", nm.ConnectionState(state).name)
            return ExitStatus.GENERIC_ERROR


def main_wrapper() -> None:
    raise SystemExit(asyncio.run(main(**get_args())))


if __name__ == "__main__":
    main_wrapper()
