import csv
import time
from datetime import datetime
from pathlib import Path

import serial


# ============================================================
# Serial configuration
# ============================================================

PORT = "COM5"
BAUDRATE = 115200
TIMEOUT = 0.2


# ============================================================
# Robot commands
# ============================================================

CMD_LEFT = "L"
CMD_RIGHT = "R"
CMD_GO = "G"
CMD_STOP = "T"


# ============================================================
# Action constants
# ============================================================

STRAIGHT = 0
LEFT = 1
RIGHT = 2


ACTION_NAME = {
    STRAIGHT: "STRAIGHT",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
}


# ============================================================
# Motion timing
# ============================================================

# Gazebo実験に近い初期値
FORWARD_TIME = 0.25
TURN_TIME = 0.08

# 制御間隔
CONTROL_INTERVAL = 0.05


# ============================================================
# Log
# ============================================================

LOG_FILE = Path("robot_run_log.csv")


def prepare_log():
    """Create CSV log file."""

    if LOG_FILE.exists():
        return

    with LOG_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "time",
            "reflection",
            "last_turn",
            "action",
        ])


def save_log(
    reflection,
    last_turn,
    action,
):
    """Save one decision."""

    with LOG_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            datetime.now().isoformat(
                timespec="milliseconds"
            ),
            reflection,
            last_turn,
            action,
        ])


# ============================================================
# Serial communication
# ============================================================

def send_command(
    ser,
    command,
):
    """Send command to robot."""

    message = f"{command}\n"

    ser.write(
        message.encode("ascii")
    )

    ser.flush()

    print(
        f"TX : {command}"
    )


def read_sensor(ser):
    """
    Expected messages:

        S,1
        S,0

    Return:
        1 : reflection
        0 : no reflection
        None : invalid message
    """

    raw = ser.readline()

    if not raw:
        return None

    text = raw.decode(
        "ascii",
        errors="ignore",
    ).strip()

    print(
        f"RX : {text}"
    )

    if text == "S,1":
        return 1

    if text == "S,0":
        return 0

    return None


# ============================================================
# Rule controller
# ============================================================

def decide_action(
    reflection,
    last_turn,
):
    """
    Same rule used in Gazebo.

    reflection = 1
        -> STRAIGHT

    reflection = 0 and previous RIGHT
        -> LEFT

    reflection = 0 and previous LEFT
        -> RIGHT
    """

    if reflection == 1:
        return STRAIGHT

    if last_turn == RIGHT:
        return LEFT

    return RIGHT


# ============================================================
# Robot movement
# ============================================================

def stop_robot(ser):

    send_command(
        ser,
        CMD_STOP,
    )


def execute_action(
    ser,
    action,
):

    if action == STRAIGHT:

        send_command(
            ser,
            CMD_GO,
        )

        time.sleep(
            FORWARD_TIME
        )

    elif action == LEFT:

        send_command(
            ser,
            CMD_LEFT,
        )

        time.sleep(
            TURN_TIME
        )

    elif action == RIGHT:

        send_command(
            ser,
            CMD_RIGHT,
        )

        time.sleep(
            TURN_TIME
        )

    stop_robot(
        ser
    )


# ============================================================
# Main
# ============================================================

def main():

    prepare_log()

    print(
        "================================"
    )

    print(
        "Robot Direction Controller"
    )

    print(
        "================================"
    )

    print(
        f"Port     : {PORT}"
    )

    print(
        f"Baudrate : {BAUDRATE}"
    )

    print()

    ser = serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=TIMEOUT,
    )

    # USBシリアル接続直後の安定待ち
    time.sleep(
        2.0
    )

    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # Gazebo実験と同じ初期値
    last_turn = RIGHT

    print(
        "Robot connected."
    )

    print(
        "Ctrl+C : stop"
    )

    print()

    try:

        # 念のため開始時停止
        stop_robot(
            ser
        )

        while True:

            # ----------------------------------------
            # Sensor
            # ----------------------------------------

            reflection = read_sensor(
                ser
            )

            if reflection is None:

                time.sleep(
                    CONTROL_INTERVAL
                )

                continue

            # ----------------------------------------
            # Decision
            # ----------------------------------------

            previous_turn = (
                last_turn
            )

            action = decide_action(
                reflection,
                previous_turn,
            )

            print(
                f"reflection={reflection}  "
                f"last_turn="
                f"{ACTION_NAME[previous_turn]}  "
                f"action="
                f"{ACTION_NAME[action]}"
            )

            # ----------------------------------------
            # Log
            # ----------------------------------------

            save_log(
                reflection,
                previous_turn,
                action,
            )

            # ----------------------------------------
            # Robot action
            # ----------------------------------------

            execute_action(
                ser,
                action,
            )

            # ----------------------------------------
            # Update history
            # ----------------------------------------

            if action == LEFT:
                last_turn = LEFT

            elif action == RIGHT:
                last_turn = RIGHT

            time.sleep(
                CONTROL_INTERVAL
            )

    except KeyboardInterrupt:

        print()
        print(
            "Stopping robot..."
        )

    finally:

        try:
            stop_robot(
                ser
            )

            time.sleep(
                0.1
            )

        finally:
            ser.close()

        print(
            "Serial port closed."
        )


if __name__ == "__main__":
    main()
