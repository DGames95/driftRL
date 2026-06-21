###


at first lambda = 0.99 used - looks too far back, discount needs to be increased.


terminal penalty was too small at start -> set to 10000, but then no progress, as all progress is just a rounding error. Printing the reward and playing the game gave an idea if we were roughly in the right ballpark.


next attempt is to add a progress reward