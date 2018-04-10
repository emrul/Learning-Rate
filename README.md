# Learning Rate

Keras Callbacks to :

- Find the optimal Learning Rate
- Use Stochastic Gradient Descent with Restart
- Use Cyclical Learning Rate

## Learning Rate Finder

### Usage 

```
lr_finder = LRFinder(min_lr=1e-5, max_lr=1e-2, steps_per_epoch=np.ceil(X_train.shape[0]/batch_size), epochs=1)

model.fit(X_train, y_train, callbacks=[lr_finder] )
```


```
# Plot the raw losses (they can be noisy)
lr_finder.plot_loss()
```

![](Images/RawLoss.png)

```
# Plot the smoothed losses
lr_finder.plot_avg_loss()
```

![](Images/SmoothLoss.png)

## Stochastic Gradient Descent with Restart

```
schedule = SGDRScheduler(min_lr=1e-4,max_lr=1e-3,steps_per_epoch=np.ceil(X_train.shape[0]/batch_size), lr_decay=0.9, cycle_length=3,mult_factor=1.5)

model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test), callbacks=[schedule])
```

![](Images/SGDR.png)

## Cyclical Learning Rate

```
clr_triangular = CyclicLR(mode='triangular', step_size=np.ceil(X_train.shape[0]/batch_size))

model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test), callbacks=[clr_triangular])
```

![](Images/CLR.png)

## References

This code is based on:

- The method described in the 2015 paper "Cyclical Learning Rates for Training Neural Networks" by Leslie N. Smith
- The implementation of the algorithm in fastai library by Jeremy Howard.
- [This](https://github.com/bckenstler/CLR) implementation of CLR

