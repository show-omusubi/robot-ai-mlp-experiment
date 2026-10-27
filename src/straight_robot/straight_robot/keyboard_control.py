import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class KeyboardControl(Node):

    def __init__(self):
        super().__init__('keyboard_control')

        self.publisher_ = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.linear_speed = 0.30
        self.angular_speed = 0.80

        self.forward_time = 0.50
        self.turn_time = 0.30

        self.get_logger().info(
            's=straight, l=left, r=right, q=quit'
        )

    def publish_velocity(self, linear, angular):
        msg = Twist()

        msg.linear.x = linear
        msg.angular.z = angular

        self.publisher_.publish(msg)

    def stop(self):
        self.publish_velocity(0.0, 0.0)

    def move_straight(self):
        self.get_logger().info('STRAIGHT')

        self.publish_velocity(
            self.linear_speed,
            0.0
        )

        time.sleep(self.forward_time)

        self.stop()

    def turn_left(self):
        self.get_logger().info('LEFT')

        self.publish_velocity(
            0.0,
            self.angular_speed
        )

        time.sleep(self.turn_time)

        self.stop()

    def turn_right(self):
        self.get_logger().info('RIGHT')

        self.publish_velocity(
            0.0,
            -self.angular_speed
        )

        time.sleep(self.turn_time)

        self.stop()


def main(args=None):

    rclpy.init(args=args)

    node = KeyboardControl()

    try:

        while rclpy.ok():

            command = input(
                'command [s/l/r/q] > '
            ).strip().lower()

            if command == 's':
                node.move_straight()

            elif command == 'l':
                node.turn_left()

            elif command == 'r':
                node.turn_right()

            elif command == 'q':
                node.stop()
                break

            else:
                print('s, l, r, q のいずれかを入力してください')

            rclpy.spin_once(
                node,
                timeout_sec=0.01
            )

    except KeyboardInterrupt:
        pass

    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
