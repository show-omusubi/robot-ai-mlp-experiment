import csv
import math
import os
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


STRAIGHT = 0
LEFT = 1
RIGHT = 2

ACTION_NAME = {
    STRAIGHT: 'STRAIGHT',
    LEFT: 'LEFT',
    RIGHT: 'RIGHT',
}


class RuleController(Node):

    def __init__(self):
        super().__init__('rule_controller')

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.subscription = self.create_subscription(
            LaserScan,
            '/front_range',
            self.scan_callback,
            10
        )

        # 最後に行った旋回方向
        self.last_turn = RIGHT

        # センサ判定
        self.threshold = 4.5

        # 動作パラメータ
        self.linear_speed = 0.20
        self.angular_speed = 0.35

        self.forward_time = 0.25
        self.turn_time = 0.08

        self.busy = False

        # CSV保存先
        self.data_file = os.path.expanduser(
            '~/robot_ai_ws/training_data.csv'
        )

        self.prepare_csv()

        self.get_logger().info(
            f'Rule controller started. '
            f'Data file: {self.data_file}'
        )

    def prepare_csv(self):
        """CSVがなければヘッダーを作る。"""

        if os.path.exists(self.data_file):
            return

        with open(
            self.data_file,
            'w',
            newline='',
            encoding='utf-8'
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                'reflection',
                'last_turn',
                'action',
            ])

    def save_data(
        self,
        reflection,
        last_turn,
        action
    ):
        """1判断分をCSVへ保存する。"""

        with open(
            self.data_file,
            'a',
            newline='',
            encoding='utf-8'
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                reflection,
                last_turn,
                action,
            ])

    def publish_velocity(
        self,
        linear,
        angular
    ):
        msg = Twist()

        msg.linear.x = linear
        msg.angular.z = angular

        self.cmd_pub.publish(msg)

    def stop(self):
        self.publish_velocity(
            0.0,
            0.0
        )

    def move_straight(self):
        self.get_logger().info(
            'ACTION = STRAIGHT'
        )

        self.publish_velocity(
            self.linear_speed,
            0.0
        )

        time.sleep(
            self.forward_time
        )

        self.stop()

    def turn_left(self):
        self.get_logger().info(
            'ACTION = LEFT'
        )

        self.publish_velocity(
            0.0,
            self.angular_speed
        )

        time.sleep(
            self.turn_time
        )

        self.stop()

        self.last_turn = LEFT

    def turn_right(self):
        self.get_logger().info(
            'ACTION = RIGHT'
        )

        self.publish_velocity(
            0.0,
            -self.angular_speed
        )

        time.sleep(
            self.turn_time
        )

        self.stop()

        self.last_turn = RIGHT

    def scan_callback(self, msg):
        """センサ値を受けてルール制御する。"""

        if self.busy:
            return

        if len(msg.ranges) == 0:
            return

        distance = msg.ranges[0]

        # 反射あり / なし判定
        if math.isinf(distance) or math.isnan(distance):
            reflection = 0

        elif distance < self.threshold:
            reflection = 1

        else:
            reflection = 0

        # 今回の判断前のlast_turnを保存
        previous_turn = self.last_turn

        # -----------------------------------------
        # ルールによる教師ラベル生成
        # -----------------------------------------

        if reflection == 1:
            action = STRAIGHT

        elif previous_turn == RIGHT:
            action = LEFT

        else:
            action = RIGHT

        # 学習データ保存
        self.save_data(
            reflection,
            previous_turn,
            action
        )

        self.get_logger().info(
            f'distance={distance:.3f} '
            f'reflection={reflection} '
            f'last_turn={ACTION_NAME[previous_turn]} '
            f'action={ACTION_NAME[action]}'
        )

        self.busy = True

        try:
            if action == STRAIGHT:
                self.move_straight()

            elif action == LEFT:
                self.turn_left()

            elif action == RIGHT:
                self.turn_right()

        finally:
            self.busy = False


def main(args=None):

    rclpy.init(args=args)

    node = RuleController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
