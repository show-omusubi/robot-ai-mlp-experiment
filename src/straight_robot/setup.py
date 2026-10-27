from setuptools import find_packages, setup

package_name = 'straight_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='show',
    maintainer_email='show@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
		'keyboard_control = straight_robot.keyboard_control:main',
        	'reflection_monitor = straight_robot.reflection_monitor:main',
		'rule_controller = straight_robot.rule_controller:main',
		'episode_collector = straight_robot.episode_collector:main',
        ],
    },
)
