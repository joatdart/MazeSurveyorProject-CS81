define lidar ranger
(
  sensor
  (
    range [0 20]
    fov 360.0
    samples 721
  )
  # generic model properties
  color "black"
)


define rosbot position
(
  #localization "gps"
  localization "odom"
  odom_error [0.0 0.01 0.0 0.01 0.0 0.01]

  size [ 0.2 0.2 0.2 ]
  origin [ 0.0 0.0 0.0 0.0 ]
  
  gui_nose 1
  drive "diff"
  color "red"
  obstacle_return 1    # Can hit things.
  laser_return 1
  
  lidar(pose [ 0.0 0.0 0.1 0.0 ])
)

define actionbot position
(
  #localization "gps"
  localization "odom"
  odom_error [0.0 0.01 0.0 0.01 0.0 0.01]

  size [ .5 .5 1 ]
  origin [ 0.0 0.0 0.0 0.0 ]
  
  gui_nose 1
  drive "diff"
  color "blue"
  obstacle_return 1    # Can hit things.
  laser_return 1
  
  lidar(pose [ 0.0 0.0 0.1 0.0 ])

  block
  (
    points 4
    point[3] [0.45 1]
    point[2] [1 1]
    point[1] [1 0]
    point[0] [0.45 0]
    z [0 1]
  )
)