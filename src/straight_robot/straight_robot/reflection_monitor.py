import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ReflectionMonitor(Node):

    def __init__(self):
        super().__init__('reflection_monitor')

        self.subscription = self.create_subscription(
            LaserScan,
            '/front_range',
            self.scan_callback,
            10
        )

        # 今回は5mが最大測距なので、
        # 4.5m未満なら反射ありとみなす
        self.threshold = 4.5

        self.get_logger().info(
            'Reflection monitor started'
        )

    def scan_callback(self, msg):

        if len(msg.ranges) == 0:
            return

        distance = msg.ranges[0]

        # inf / nan の場合は反射なし
        if math.isinf(distance) or math.isnan(distance):
            reflection = 0

        elif distance < self.threshold:
            reflection = 1

        else:
            reflection = 0

        self.get_logger().info(
            f'distance={distance:.3f} m  reflection={reflection}'
        )


def main(args=None):

    rclpy.init(args=args)

    node = ReflectionMonitor()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
