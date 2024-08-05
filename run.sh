#!/bin/bash
if [[ $USER != "fs" ]]; then
    export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
fi

mkdir -p logs/

tasks=(
# "close_box"
# "empty_dishwasher"
# "open_drawer"
# "slide_block_to_target"
# "stack_cups"
# "put_shoes_in_box"
# "take_shoes_out_of_box"
# "basketball_in_hoop"
# "set_the_table"
# "put_books_on_bookshelf"
# "put_bottle_in_fridge"
# "place_cups"
# "empty_container"
# "put_plate_in_colored_dish_rack"
# "put_item_in_drawer"
# "take_item_out_of_drawer"
# "screw_nail"
# "stack_chairs"
# "hockey"
# "put_knife_on_chopping_board"
# "put_toilet_roll_on_stand"
# "hang_frame_on_hanger"
# "stack_wine"
# "put_knife_in_knife_block"
# "take_frame_off_hanger"
# "get_ice_from_fridge"
# "meat_off_grill"
# "meat_on_grill"
# "phone_on_base"
# "put_all_groceries_in_cupboard"
# "put_groceries_in_cupboard"
# "put_money_in_safe"
# "put_rubbish_in_bin"
# "setup_checkers"
# "setup_chess"
# "solve_puzzle"

## New tasks on 0711
# "move_hanger"
# "place_hanger_on_rack"
# "pour_from_cup_to_cup"
# "put_tray_in_oven"
# "put_umbrella_in_umbrella_stand"
# "remove_cups"
# "stack_blocks"
# "take_tray_out_of_oven"
# "toilet_seat_up"
# "turn_tap"
# "tv_on"
# "water_plants"
# "weighing_scales"

## New tasks on 0726
"beat_the_buzz"
"block_pyramid"
"change_channel"
"change_clock"
"close_door"
"close_drawer"
"close_fridge"
"close_grill"
"close_jar"
"close_laptop_lid"
"close_microwave"
"hit_ball_with_queue"
"insert_onto_square_peg"
"insert_usb_in_computer"
"lamp_off"
"lamp_on"
"lift_numbered_block"
"light_bulb_in"
"light_bulb_out"
"open_box"
"open_door"
"open_fridge"
"open_grill"
"open_jar"
"open_microwave"
"open_oven"
"open_washing_machine"
"open_window"
"open_wine_bottle"
"pick_and_lift"
"pick_and_lift_small"
"pick_up_cup"
"place_shape_in_shape_sorter"
"play_jenga"
"plug_charger_in_power_supply"
"press_switch"
"push_button"
"push_buttons"
"reach_and_drag"
"reach_target"
"scoop_with_spatula"
"slide_cabinet_open_and_place_cups"
"straighten_rope"
"sweep_to_dustpan"
"take_cup_out_from_cabinet"
"take_lid_off_saucepan"
"take_off_weighing_scales"
"take_plate_off_colored_dish_rack"
"take_toilet_roll_off_stand"
"take_umbrella_out_of_umbrella_stand"
"take_usb_out_of_computer"
"toilet_seat_down"
"turn_oven_on"
"unplug_charger"
"wipe_desk"
)

for task in ${tasks[@]}; do
    ## Record object states
    # python tools/collect_demo.py --episode_num=500 --headless --task=$task --record_object_states

    ## Render
    # nohup python tools/replay_object_states.py --headless --task $task > logs/$task.out &
    # python tools/replay_object_states.py --headless --task $task 2>&1 | tee logs/$task.out

    ## Test
    python tools/collect_demo.py --episode_num=1 --headless --task=$task --record_object_states | tee logs/$task.out
    python tools/replay_object_states.py --headless --task $task 2>&1 --max_demo 1 --overwrite | tee -a logs/$task.out
done
