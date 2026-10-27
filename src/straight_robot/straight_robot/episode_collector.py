import csv
import math
import os
import random
import subprocess
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


class EpisodeCollector(Node):

    def __init__(self):
        super().__init__('episode_collector')

        # -------------------------------------------------
        # ROS
        # -------------------------------------------------
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/front_range',
            self.scan_callback,
            10
        )

        # -------------------------------------------------
        # 実験条件
        # -------------------------------------------------
        self.num_episodes = 60

        # 1エピソードの走行時間
        self.episode_duration = 3.0

        # リセット後の待ち時間
        self.settle_time = 0.5

        # 制御周期
        self.control_period = 0.10

        # センサ
        self.reflection_threshold = 4.5

        # 動作
        self.linear_speed = 0.20
        self.angular_speed = 0.35

        # 反射板の中心
        self.target_x = 2.0
        self.target_y = 0.0

        # 再現可能な乱数
        self.random = random.Random(42)

        # -------------------------------------------------
        # 状態
        # -------------------------------------------------
        self.latest_reflection = None

        self.episode = -1
        self.sample_number = 0

        self.last_turn = RIGHT

        self.initial_x = 0.0
        self.initial_y = 0.0
        self.initial_yaw = 0.0

        self.episode_start_time = None
        self.settle_until = None

        self.state = 'RESET'
        self.finished = False

        # -------------------------------------------------
        # CSV
        # -------------------------------------------------
        self.data_file = os.path.expanduser(
            '~/robot_ai_ws/episode_training_data.csv'
        )

        self.prepare_csv()

        # 10 Hz
        self.timer = self.create_timer(
            self.control_period,
            self.control_callback
        )

        self.get_logger().info(
            f'Automatic experiment started: '
            f'{self.num_episodes} episodes'
        )

    # =====================================================
    # CSV
    # =====================================================

    def prepare_csv(self):

        with open(
            self.data_file,
            'w',
            newline='',
            encoding='utf-8'
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                'episode',
                'sample',
                'initial_x',
                'initial_y',
                'initial_yaw_deg',
                'reflection',
                'last_turn',
                'action',
            ])

    def save_data(
        self,
        reflection,
        previous_turn,
        action
    ):

        with open(
            self.data_file,
            'a',
            newline='',
            encoding='utf-8'
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                self.episode,
                self.sample_number,
                self.initial_x,
                self.initial_y,
                math.degrees(self.initial_yaw),
                reflection,
                previous_turn,
                action,
            ])

    # =====================================================
    # Sensor
    # =====================================================

    def scan_callback(self, msg):

        if len(msg.ranges) == 0:
            return

        distance = msg.ranges[0]

        if math.isinf(distance) or math.isnan(distance):
            self.latest_reflection = 0

        elif distance < self.reflection_threshold:
            self.latest_reflection = 1

        else:
            self.latest_reflection = 0

    # =====================================================
    # Robot command
    # =====================================================

    def publish_velocity(
        self,
        linear,
        angular
    ):

        msg = Twist()

        msg.linear.x = linear
        msg.angular.z = angular

        self.cmd_pub.publish(msg)

    def stop_robot(self):

        self.publish_velocity(
            0.0,
            0.0
        )

    # =====================================================
    # Gazebo reset
    # =====================================================

    def set_robot_pose(
        self,
        x,
        y,
        yaw
    ):

        # yaw -> quaternion
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        request = (
            f'name: "straight_robot", '
            f'position: {{x: {x}, y: {y}, z: 0.0}}, '
            f'orientation: {{x: 0.0, y: 0.0, '
            f'z: {qz}, w: {qw}}}'
        )

        command = [
            'gz',
            'service',
            '-s',
            '/world/straight_test/set_pose',
            '--reqtype',
            'gz.msgs.Pose',
            '--reptype',
            'gz.msgs.Boolean',
            '--timeout',
            '3000',
            '--req',
            request,
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            self.get_logger().error(
                f'set_pose failed: {result.stderr}'
            )

            return False

        return True

    # =====================================================
    # New episode
    # =====================================================

    def start_new_episode(self):

        self.episode += 1

        if self.episode >= self.num_episodes:

            self.stop_robot()

            self.finished = True

            self.timer.cancel()

            self.get_logger().info(
                'All episodes completed.'
            )

            self.get_logger().info(
                f'Data saved: {self.data_file}'
            )

            return

        # ---------------------------------------------
        # 初期条件を自動生成
        # ---------------------------------------------

        self.initial_x = self.random.uniform(
            -0.20,
            0.20
        )

        self.initial_y = self.random.uniform(
            -0.15,
            0.15
        )

        yaw_deg = self.random.uniform(
            -12.0,
            12.0
        )

        self.initial_yaw = math.radians(
            yaw_deg
        )

        # ---------------------------------------------
        # 目標方向を計算
        # ---------------------------------------------

        desired_yaw = math.atan2(
            self.target_y - self.initial_y,
            self.target_x - self.initial_x
        )

        error = (
            desired_yaw
            - self.initial_yaw
        )

        # -pi ～ +pi
        error = math.atan2(
            math.sin(error),
            math.cos(error)
        )

        # ---------------------------------------------
        # last_turn初期値
        #
        # targetが左側なら、次はLEFTが必要
        # → last_turnはRIGHTとする
        #
        # targetが右側なら逆
        # ---------------------------------------------

        if error > 0.0:
            self.last_turn = RIGHT

        elif error < 0.0:
            self.last_turn = LEFT

        else:
            self.last_turn = RIGHT

        self.stop_robot()

        success = self.set_robot_pose(
            self.initial_x,
            self.initial_y,
            self.initial_yaw
        )

        if not success:
            self.finished = True
            return

        self.sample_number = 0
        self.latest_reflection = None

        self.settle_until = (
            time.monotonic()
            + self.settle_time
        )

        self.state = 'SETTLING'

        self.get_logger().info(
            f'EP {self.episode:03d}: '
            f'x={self.initial_x:+.3f}, '
            f'y={self.initial_y:+.3f}, '
            f'yaw={yaw_deg:+.1f} deg'
        )

    # =====================================================
    # Control loop
    # =====================================================

    def control_callback(self):

        if self.finished:
            return

        # ---------------------------------------------
        # 最初または次エピソード
        # ---------------------------------------------

        if self.state == 'RESET':

            self.start_new_episode()

            return

        # ---------------------------------------------
        # リセット後の安定待ち
        # ---------------------------------------------

        if self.state == 'SETTLING':

            self.stop_robot()

            if time.monotonic() >= self.settle_until:

                self.episode_start_time = (
                    time.monotonic()
                )

                self.state = 'RUNNING'

            return

        # ---------------------------------------------
        # 走行
        # ---------------------------------------------

        if self.state != 'RUNNING':
            return

        elapsed = (
            time.monotonic()
            - self.episode_start_time
        )

        # エピソード終了
        if elapsed >= self.episode_duration:

            self.stop_robot()

            self.state = 'RESET'

            return

        # センサ待ち
        if self.latest_reflection is None:
            return

        reflection = self.latest_reflection

        previous_turn = self.last_turn

        # ---------------------------------------------
        # 教師ルール
        # ---------------------------------------------

        if reflection == 1:

            action = STRAIGHT

        elif previous_turn == RIGHT:

            action = LEFT

        else:

            action = RIGHT

        # ---------------------------------------------
        # 保存
        # ---------------------------------------------

        self.save_data(
            reflection,
            previous_turn,
            action
        )

        self.sample_number += 1

        # ---------------------------------------------
        # 動作
        # ---------------------------------------------

        if action == STRAIGHT:

            self.publish_velocity(
                self.linear_speed,
                0.0
            )

        elif action == LEFT:

            self.publish_velocity(
                0.0,
                self.angular_speed
            )

            self.last_turn = LEFT

        elif action == RIGHT:

            self.publish_velocity(
                0.0,
                -self.angular_speed
            )

            self.last_turn = RIGHT


def main(args=None):

    rclpy.init(args=args)

    node = EpisodeCollector()

    try:

        while (
            rclpy.ok()
            and not node.finished
        ):

            rclpy.spin_once(
                node,
                timeout_sec=0.1
            )

    except KeyboardInterrupt:
        pass

    finally:

        node.stop_robot()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
