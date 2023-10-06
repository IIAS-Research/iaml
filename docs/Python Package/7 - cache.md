# Cache

Pipeline Steps use cache to optimize treatment time.
The caching method is really simple : for the same configuration and input, a step will have the same output.

Cache is fully automatized but sometimes this is not desirable. So you have several way to impact the cache behavior :

- Reset the cache -> `step.reset_cache()`
- disable or enable the cache
    - With the constructor -> you can give a boolean value to the parameter "use_cache". Default is True.
    - With the use_cache setter -> `step.use_cache = False`


# In your own step 

If you create your own Step and want to change the cache behavior, you can for example :

- Change the default value of use_cache in the constructor
- Override the use_cache setter to avoid value to be edited 
- It's your own class, be inventive ! ;)