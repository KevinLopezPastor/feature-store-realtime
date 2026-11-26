"""Feast entities: the keys features are looked up by."""

from feast import Entity
from feast.value_type import ValueType

# Pickup zone (TLC PULocationID, 1 to 265)
pickup_zone = Entity(
    name="pickup_zone",
    join_keys=["PULocationID"],
    value_type=ValueType.INT32,
    description="TLC Taxi Zone PULocationID",
)

# Hour of week (0 to 167)
# 0 represents Monday 00:00 NYC local time
hour_of_week = Entity(
    name="hour_of_week",
    join_keys=["hour_of_week"],
    value_type=ValueType.INT32,
    description="Hour of the week (0-167, starting Monday 00:00 local)",
)
