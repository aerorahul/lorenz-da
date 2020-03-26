__all__ = ['ModelBase']

class ModelBase(object):
    '''
    This provides a base class for all models.
    '''

    def __init__(self, Name, dt):
        '''
        Populates the basics of a model class such as Name and time-step
        '''
        self.Name = Name
        self.dt = dt

    def __repr__(self):
        return {'Name':self.Name, 'dt':self.dt}

    def __str__(self):
        return 'ModelBase(Name='+self.Name+', dt='+str(self.dt)+ ')'

    def advance(self, modelFunc, x0, t, *args, **kwargs):
        '''
        method that integrates the model state x0(t0) -> xt(T), using the model advance
        function 'modelFunc', initial conditions 'x0' and length of the integration in 't'.
        Additional arguments are provided via args and kwargs.

        xt = advance(self, modelFunc, x0, t, *args, **kwargs)

    modelFunc - model advance function to call
           x0 - initial state at time t = t0
            t - vector of time from t = [t0, T]
        *args - any additional arguments
     **kwargs - any additional keyword arguments
           xt - final state at time t = T
        '''

        # Let the specific model implement the advance method
        print('ModelBase does not implement the advance method.')
        return NotImplemented
