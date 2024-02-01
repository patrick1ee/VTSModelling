function [chan_idx] = get_chan_idx(Const, chan_name)

chan_idx = find(strcmp(Const.chan_order.chan_name, chan_name));   
