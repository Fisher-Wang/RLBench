#!/bin/bash

EPISODE_NUM=10
tasks=(
    # close_box
    # put_shoes_in_box
    # take_shoes_out_of_box
    # basketball_in_hoop
    # set_the_table
    # empty_dishwasher
    # put_books_on_bookshelf
    # put_bottle_in_fridge
    # place_cups
    # take_item_out_of_drawer
    # put_item_in_drawer
    # stack_wine
    # take_frame_off_hanger
    # put_knife_in_knife_block
    # place_cups
    # put_knife_on_chopping_board
    # put_toilet_roll_on_stand
    # hockey
    # stack_chairs

    ## New pick&place only
    get_ice_from_fridge
    meat_off_grill
    meat_on_grill
    phone_on_base
    put_all_groceries_in_cupboard
    put_groceries_in_cupboard
    put_money_in_safe
    put_rubbish_in_bin
    setup_checkers
    setup_chess
    solve_puzzle
)

# Start the YAML configuration file
echo "name: tasks"
echo "root: .  # Adjust this to your project's root directory if necessary"
echo ""
echo "# Define a separate window for each task, each running in its own session"
echo "windows:"

# Loop through each task and generate the corresponding YAML structure
for task in "${tasks[@]}"; do
    echo "  - $task:"
    echo "      panes:"
    echo "        - python tools/collect_demo.py --episode_num=$EPISODE_NUM --headless --task=$task"
done