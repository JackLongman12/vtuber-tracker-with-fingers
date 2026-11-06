"""Ask Windows to treat this process as real-time work.

Windows 11 on hybrid CPUs (fast + power-saving cores) moves programs that aren't the focused window onto the
slow cores and lowers their clock ("power throttling"). The studio is almost never the focused window (your
VTuber app or game is), so it opts this one process out and raises its priority a step. Nothing system-wide
is changed; it ends when the studio closes.

macOS does the same with "App Nap" (apps not in front get their timers and CPU slowed): the process declares
latency-critical work instead, which keeps App Nap away but still lets the Mac sleep when idle. Linux doesn't
throttle background programs like this."""
import ctypes
import sys

_mac_activity = None  # (macOS: the "busy" declaration lasts while this is kept)


def _mac_no_app_nap():
    global _mac_activity
    try:
        import Foundation  # (pyobjc, which pywebview brings along on macOS)
    except ImportError:
        return "App Nap stays on (pyobjc missing)"
    options = getattr(Foundation, "NSActivityUserInitiatedAllowingIdleSystemSleep", 0x00EFFFFF) | \
        getattr(Foundation, "NSActivityLatencyCritical", 0xFF00000000)
    _mac_activity = Foundation.NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
        options, "Avatar Studio: live tracking")
    return "App Nap off, latency-critical"


class _ThrottleState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def boost_process():
    """Returns a short description of what was applied (for the startup log)."""
    if sys.platform == "darwin":
        return _mac_no_app_nap()
    if sys.platform != "win32":
        return "not Windows or macOS: nothing to do"
    done = []
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p  # a HANDLE, not an int
    k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    proc = k32.GetCurrentProcess()
    # ProcessPowerThrottling (4): control EXECUTION_SPEED (1), state 0 = never throttle this process
    state = _ThrottleState(1, 1, 0)
    if k32.SetProcessInformation(proc, 4, ctypes.byref(state), ctypes.sizeof(state)):
        done.append("no power throttling")
    if k32.SetPriorityClass(proc, 0x00008000):  # ABOVE_NORMAL_PRIORITY_CLASS
        done.append("above-normal priority")
    if ctypes.windll.winmm.timeBeginPeriod(1) == 0:  # 1 ms timer: sleeps and waits wake up on time
        done.append("1 ms timer")
    return ", ".join(done) or "not applied"
