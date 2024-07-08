export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

mkdir -p logs/

# for task in close_box empty_dishwasher open_drawer slide_block_to_target stack_cups put_shoes_in_box take_shoes_out_of_box basketball_in_hoop screw_nail;

# for task in put_toilet_roll_on_stand hang_frame_on_hanger stack_wine put_knife_in_knife_block take_frame_off_hanger get_ice_from_fridge meat_off_grill meat_on_grill hockey;
# for task in phone_on_base put_all_groceries_in_cupboard in put_bottle_in_fridge place_cups empty_container put_plate_in_colored_dish_rack put_item_in_drawer take_item_out_of_drawer put_knife_on_chopping_board;
# for task in set_the_table put_books_on_bookshelf put_groceries_in_cupboard put_money_in_safe put_rubbish_in_bin setup_checkers setup_chess solve_puzzle stack_chairs;

for task in meat_off_grill meat_on_grill hockey;
do
    # python tools/collect_demo.py --episode_num=500 --headless --task=$task --record_object_states
    nohup python tools/replay_object_states.py --headless --task $task > logs/$task.out &
done
