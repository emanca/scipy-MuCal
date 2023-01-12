import os
import multiprocessing
import functools
import itertools

#ncpu = multiprocessing.cpu_count()

#os.environ["OMP_NUM_THREADS"] = str(ncpu)
#os.environ["OPENBLAS_NUM_THREADS"] = str(ncpu)
#os.environ["MKL_NUM_THREADS"] = str(ncpu)
#os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncpu)
#os.environ["NUMEXPR_NUM_THREADS"] = str(ncpu)

import jax
import jax.numpy as np
import numpy as onp
from jax import grad, hessian, jacobian, config
from jax.scipy.special import erf

config.update('jax_enable_x64', True)

def hessianlowmem(fun):
    def _hessianlowmem(x, *args):
        def funp(x):
            return fun(x,*args)
        _, hvp = jax.linearize(jax.grad(funp), x)
        #hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        return jax.lax.map(hvp,basis)
        #return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return _hessianlowmem

def hessianlowmemb(fun, batch_size=8):
    def _hessianlowmem(x, *args):
        def funp(x):
            return fun(x,*args)
        _, hvp = jax.linearize(jax.grad(funp), x)
        #hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        return batch_vmap(hvp,batch_size=batch_size)(basis)
        #return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return _hessianlowmem

def jaclowmemb(fun, batch_size=8):
    def _jaclowmem(x, *args):
        def funp(x):
            return fun(x,*args)
        _, jvp = jax.linearize(funp, x)
        #hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        return batch_vmap(jvp,batch_size=batch_size)(basis)
        #return batch_vmap.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return _jaclowmem

def jacvlowmemb(fun, batch_size=8):
    def _hessianlowmem(x, *args):
        def funp(x):
            return fun(x,*args)
        #_, hvp = jax.linearize(jax.grad(funp), x)
        def jvp(v):
            return jax.jvp(funp,(x,),(v,))[1]
        #hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        #jac = jax.lax.map(jvp,basis)
        jac = batch_vmap(jvp, batch_size=batch_size)(basis)
        #hess = np.swapaxes(jac,0,1)
        return jac
        #return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return _hessianlowmem

def jaclowmem(fun):
    def _jaclowmem(x, *args):
        def funp(x):
            return fun(x,*args)
        _, jvp = jax.linearize(funp, x)
        #hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        return jax.lax.map(jvp,basis)
        #return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return _jaclowmem



def hessianvlowmem(fun):
    def _hessianlowmem(x, *args):
        def funp(x):
            return fun(x,*args)
        #_, hvp = jax.linearize(jax.grad(funp), x)
        def gdotv(x,v):
            return np.dot(jax.grad(funp)(x),v)
        def hvp(v):
            return jax.grad(gdotv)(x,v)
        #hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        return jax.lax.map(hvp,basis)
        #return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return _hessianlowmem


def hessianvlowmem2(fun):
    def _hessianlowmem(x, *args):
        def funp(x):
            return fun(x,*args)
        #_, hvp = jax.linearize(jax.grad(funp), x)
        def jvp(v):
            return jax.jvp(jax.grad(funp),(x,),(v,))[1]
        #hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        jac = jax.lax.map(jvp,basis)
        hess = np.swapaxes(jac,0,1)
        return hess
        #return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return _hessianlowmem


def hessianvlowmem3(fun):
    def _hessianlowmem(x, *args):
        def funp(x):
            return fun(x,*args)
        #_, hvp = jax.linearize(jax.grad(funp), x)
        def jvp(v):
            return jax.vjp(jax.grad(funp),v)[1](x)
        #hvp = jax.jit(hvp)  # seems like a substantial speedup to do this
        basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
        jac = jax.lax.map(jvp,basis)[0]
        #print(jac.shape)
        #hess = np.swapaxes(jac,0,1)
        return jac
        #return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)
    return _hessianlowmem

#def vmap(f, chunksize=128):
    #vf = jax.vmap(f)
    

#parallel minimization using orthogonal basis subspaces based on arXiv:1506.07222,
#ported from tensorflow implementation in combinetf, but skipping all of the SR1 update
#parts since the number of parameters in each fit is small and hessian+eigen-decomposition
#can be explicitly computed

#f should be a scalar valued function, x is the parameter vector to be minimized
#x and any additional arguments should have an outer batch dimension over which fits will
#be parallelized, which could correspond to e.g. independent categories/bins, or sets of data

#if doParallel is False this can be used as a standard minimizer, where x is just the parameter vector as usual

def beval2(f, batch_size=128, accumulator=lambda x: np.add(x,axis=0)):
    def _beval(*args):
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]
        idxstart = np.arange(0,length,batch_size)
        idxend = idxstart + batch_size

        out_shape = jax.eval_shape(f,*args)
        _,out_tree = jax.tree_flatten(out_shape)
        
        out_flat_batches = []
        for istart,iend in zip(idxstart,idxend):
            args_batch = []
            for arg in args:
                args_batch.append(arg[istart:iend])
            out_flat_batch, _ = jax.tree_flatten(f(*args_batch))
            out_flat_batches.append(out_flat_batch)

        out_flat_batches = [list(x) for x in zip(*out_flat_batches)]
        out_flat = []
        for out in out_flat_batches:
            out_flat.append(accumulator(out))
        out = jax.tree_util.tree_unflatten(out_tree, out_flat)
        return out
    return _beval

def beval3(f, batch_size=128, accumulator=lambda x: np.add(x,axis=0)):
    def _beval(*args):
        #args_tree = jax.tree_structure(*args)
        out_shape = jax.eval_shape(f,*args)
        out_tree = jax.tree_util.tree_structure(out_shape)
        
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]
        
        splitidx = np.arange(batch_size, length, batch_size)
        splitargs_flat = map(lambda x: np.split(x,splitidx), args_flat)
        split_tree = jax.tree_util.tree_structure(splitargs_flat[0])
        
        splitargs = jax.tree_util.tree_transpose(split_tree, args_tree)
        splitout = itertools.starmap(f,splitargs)
        
        
        
        #splitargs = [jax.tree_unflatten(args_tree, splitarg) for splitarg in zip(*splitargs_flat)]
        
        #out = jax.tree_util.tree_unflatten(out_tree, splitout)
        
        splitout_flat,_ = jax.tree_util.tree_flatten(splitout)
        
        out_flat_batches = []
        for istart,iend in zip(idxstart,idxend):
            args_batch = []
            for arg in args:
                args_batch.append(arg[istart:iend])
            out_flat_batch, _ = jax.tree_flatten(f(*args_batch))
            out_flat_batches.append(out_flat_batch)

        out_flat_batches = [list(x) for x in zip(*out_flat_batches)]
        out_flat = []
        for out in out_flat_batches:
            out_flat.append(accumulator(out))
        out = jax.tree_util.tree_unflatten(out_tree, out_flat)
        return out
    return _beval

def batch_vmap(f, batch_size=128):
    fbatch = jax.vmap(f)
    def _beval(*args):
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]
        
        nbatchfull = length//batch_size
        batched_length = nbatchfull*batch_size
        remainder = length%batch_size
        
        args_batched = jax.tree_util.tree_map(lambda x: np.reshape(x[:batched_length], (nbatchfull, batch_size, *x.shape[1:])), args)
        out_batched = jax.lax.map(lambda x: fbatch(*x), args_batched)
        out = jax.tree_util.tree_map(lambda x: np.reshape(x, (batched_length,*x.shape[2:])), out_batched)
        
        if remainder>0:
            args_remainder = jax.tree_util.tree_map(lambda x: x[batched_length:], args)
            out_remainder = fbatch(*args_remainder)
            
            out_flat, out_tree = jax.tree_util.tree_flatten(out)
            out_remainder_flat,_ = jax.tree_util.tree_flatten(out_remainder)
            
            out_flat = [np.concatenate((out_i,out_remainder_i),axis=0) for out_i, out_remainder_i in zip(out_flat,out_remainder_flat)]
            out = jax.tree_util.tree_unflatten(out_tree, out_flat)

        return out
    return _beval

def batch_accumulate(f, batch_size=128, accumulate=lambda y,x: np.add(y,x)):
    def _beval(*args):
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]

        nbatchfull = length//batch_size
        batched_length = nbatchfull*batch_size
        remainder = length%batch_size
        
        print("nbatchfull", nbatchfull)
        print("batched_length", batched_length)
        print("remainder", remainder)
        

        args_batched = jax.tree_util.tree_map(lambda x: np.reshape(x[:batched_length], (nbatchfull, batch_size, *x.shape[1:])), args)
        
        args_first = jax.tree_util.tree_map(lambda x: x[0], args_batched)
        out_shape = jax.eval_shape(f,*args_first)
        out_zeros = jax.tree_util.tree_map(np.zeros_like, out_shape)

        def f_scan(y,x):
            print(x[0].shape)
            fx = f(*x)
            y = jax.tree_util.tree_multimap(accumulate, y, fx)
            return y,()
        f_scan = jax.jit(f_scan)
        out = jax.lax.scan(f_scan, init=out_zeros, xs=args_batched)[0]

        if remainder>0:
            args_remainder = jax.tree_util.tree_map(lambda x: x[batched_length:], args)
            print(args_remainder[0].shape)
            out_remainder = f(*args_remainder)
            out = jax.tree_util.tree_multimap(accumulate, out, out_remainder)
        return out
    return _beval


def pbatch_accumulate(f, batch_size=128, ncpu=32, in_axes = 0, accumulate=lambda y,x: np.add(y,x)):

    pf = jax.jit(jax.pmap(f, in_axes=in_axes, devices=jax.devices()[:ncpu]))

    def slice_axis(x, axis, start, end):
        if axis is None:
            return x
        else:
            slc = [slice(None)] * len(x.shape)
            slc[axis] = slice(start,end)
            return x[tuple(slc)]
        
    def batch_axis(x, axis, nbatchouter, nbatchinner=None):
        if axis is None:
            return x
        elif axis==0:
            shape = (nbatchouter,)
            if nbatchinner is not None:
                shape += nbatchinner
            shape += (x.shape[0]//np.prod(shape), *x.shape[1:])
            return np.reshape(x, shape)
        else:
            #TODO, handle axis!=0 case
            raise ValueError

    @jax.jit
    def _pbeval(*args_pbatched, out_shape):
        args_flat, args_tree = jax.tree_util.tree_flatten(args_pbatched)

        is_arg_batched = jax.tree_util.tree_multimap(lambda x,axis: False if axis is None else True, args, in_axes)
        print("is_arg_batched", is_arg_batched)
        is_arg_batched_flat = jax.tree_util.tree_leaves(is_arg_batched)
        print("is_arg_batched_flat", is_arg_batched_flat)
        mapped_idxs = []
        for iarg,isbatched in enumerate(is_arg_batched_flat):
            if isbatched:
                mapped_idxs.append(iarg)
                
        print("mapped_idxs", mapped_idxs)
                
        #args_batched_flat, args_tree = jax.tree_util.tree_flatten(args_batched)
        #TODO combine with earlier multimap
        args_mapped = jax.tree_util.tree_multimap(lambda arg,axis: None if axis is None else arg, args_batched,in_axes)
        args_mapped_flat = jax.tree_util.tree_leaves(args_mapped)
        
        #args_first = jax.tree_util.tree_map(lambda x: x[0], args_batched)
        #args_first = jax.tree_util.tree_multimap(lambda x,axis: x if axis is None else np.squeeze(slice_axis(x,axis,None,1),axis), args_batched, in_axes)
        #out_shape = jax.eval_shape(f,*args_first)
        out_shape = jax.eval_shape(f,*args_first)
        out_zeros = jax.tree_util.tree_map(np.zeros_like, out_shape)

        def f_scan(y,x):
            x_flat = list(args_flat)
            for idx,mapped_arg in zip(mapped_idxs,x):
                x_flat[idx] = mapped_arg
            x_tree = jax.tree_util.tree_unflatten(args_tree,x_flat)
            #fx = f(*x)
            fx = f(*x_tree)
            y = jax.tree_util.tree_multimap(accumulate, y, fx)
            return y,()
        #f_scan = jax.jit(f_scan)
        #out = jax.lax.scan(f_scan, init=out_zeros, xs=args_batched)[0]
        out = jax.lax.scan(f_scan, init=out_zeros, xs=args_mapped_flat)[0]

    @jax.jit
    def _peval(*args_p):
        val = pf(*args_p)
        return jax.tree_util.tree_multimap(lambda x,axis: x if axis is None else np.sum(x,axis=0), args_p, in_axes)


    def _func(*args):
        lengths = jax.tree_util.tree_multimap(lambda arg, axis: None if axis is None else arg.shape[axis], args, in_axes)
        length = jax.tree_util.tree_leaves(lengths)[0]
        
        pbatch_size = ncpu*batch_size
        npbatchfull = length//pbatch_size
        pbatched_length = npbatchfull*pbatch_size
        pbatched_remainder = length%pbatch_size
        
        val = None
        if npbatchfull>0:
            args_pbatched = jax.tree_util.tree_multimap(lambda x,axis: batch_axis(slice_axis(x,axis,None,pbatched_length),axis,ncpu,npbatchfull//ncpu), args, in_axes)
            pbatched_val = _pbeval(*args_pbatched, *args)
            
            val = pbatched_val if val is None else jax.tree_util.tree_multimap(accumulate, val, pbatched_val)
        if pbatched_remainder>0:
            npfull = pbatched_remainder//ncpu
            p_length = npfull*ncpu
            p_remainder = p_length%ncpu
            
            if npfull>0:
                args_p = jax.tree_util.tree_multimap(lambda x,axis: batch_axis(slice_axis(x,axis,pbatched_length,pbatched_length+p_length),axis,ncpu), args, in_axes)
                p_val = _peval(*args_p)
                
                val = p_val if val is None else jax.tree_util.tree_multimap(accumulate, val, p_val)
                
            if p_remainder>0:
                args_r = jax.tree_util.tree_multimap(lambda x,axis: slice_axis(x,axis,pbatched_length+p_length,None), args, in_axes)
                r_val = f(*args_r)
                val = p_val if val is None else jax.tree_util.tree_multimap(accumulate, val, r_val)
        
        return val
    
    return _func

def pbatch_accumulate8(f, batch_size=128, ncpu=32, in_axes = 0, accumulate=lambda y,x: np.add(y,x)):
    
    #@jax.jit
    def slice_axis(x, axis, start, end):
        if axis is None:
            return x
        else:
            slc = [slice(None)] * len(x.shape)
            slc[axis] = slice(start,end)
            return x[tuple(slc)]
        
    #@jax.jit
    def batch_axis(x, axis, nbatch):
        if axis is None:
            return x
        elif axis==0:
            return np.reshape(x, (nbatch, x.shape[0]//nbatch) + x.shape[1:])
        else:
            #TODO, handle axis!=0 case
            raise ValueError
    
    @jax.jit
    def _beval(*args):
        print("beval args types")
        for arg in args:
            print(type(arg))
        
        lengths = jax.tree_util.tree_multimap(lambda arg, axis: None if axis is None else arg.shape[axis], args, in_axes)
        length = jax.tree_util.tree_leaves(lengths)[0]

        nbatchfull = length//batch_size
        batched_length = nbatchfull*batch_size
        remainder = length%batch_size

        if nbatchfull>0:
            print("scanning")
            #args_batched = jax.tree_util.tree_map(lambda x: np.reshape(x[:batched_length], (nbatchfull, batch_size, *x.shape[1:])), args)
            args_batched = jax.tree_util.tree_multimap(lambda x,axis: batch_axis(slice_axis(x,axis,None,batched_length),axis,nbatchfull), args, in_axes)
            
            args_flat, args_tree = jax.tree_util.tree_flatten(args)
            
            is_arg_batched = jax.tree_util.tree_multimap(lambda x,axis: False if axis is None else True, args, in_axes)
            print("is_arg_batched", is_arg_batched)
            is_arg_batched_flat = jax.tree_util.tree_leaves(is_arg_batched)
            print("is_arg_batched_flat", is_arg_batched_flat)
            mapped_idxs = []
            for iarg,isbatched in enumerate(is_arg_batched_flat):
                if isbatched:
                    mapped_idxs.append(iarg)
                    
            print("mapped_idxs", mapped_idxs)
                    
            #args_batched_flat, args_tree = jax.tree_util.tree_flatten(args_batched)
            #TODO combine with earlier multimap
            args_mapped = jax.tree_util.tree_multimap(lambda arg,axis: None if axis is None else arg, args_batched,in_axes)
            args_mapped_flat = jax.tree_util.tree_leaves(args_mapped)
            
            #args_first = jax.tree_util.tree_map(lambda x: x[0], args_batched)
            args_first = jax.tree_util.tree_multimap(lambda x,axis: x if axis is None else np.squeeze(slice_axis(x,axis,None,1),axis), args_batched, in_axes)
            out_shape = jax.eval_shape(f,*args_first)
            out_zeros = jax.tree_util.tree_map(np.zeros_like, out_shape)

            def f_scan(y,x):
                x_flat = list(args_flat)
                for idx,mapped_arg in zip(mapped_idxs,x):
                    x_flat[idx] = mapped_arg
                x_tree = jax.tree_util.tree_unflatten(args_tree,x_flat)
                #fx = f(*x)
                fx = f(*x_tree)
                y = jax.tree_util.tree_multimap(accumulate, y, fx)
                return y,()
            #f_scan = jax.jit(f_scan)
            #out = jax.lax.scan(f_scan, init=out_zeros, xs=args_batched)[0]
            out = jax.lax.scan(f_scan, init=out_zeros, xs=args_mapped_flat)[0]
        else:
            print("not scanning")
            out = None

        if remainder>0:
            #args_remainder = jax.tree_util.tree_map(lambda x: x[batched_length:], args)
            args_remainder = jax.tree_util.tree_multimap(lambda x,axis: slice_axis(x,axis,batched_length,None), args, in_axes)
            #print("parallel args remainder shape")
            #for arg in args_remainder:
                #print(arg.shape)
            
            out_remainder = f(*args_remainder)
            if out is None:
                out = out_remainder
            else:
                out = jax.tree_util.tree_multimap(accumulate, out, out_remainder)
        return out
    
    _pbeval = jax.pmap(_beval, in_axes=in_axes, devices=jax.devices()[:ncpu])
    _pbeval = jax.jit(_pbeval)
    
    #@jax.jit
    def pbatch(*args):
        #jax.tree_util.tree_map(lambda x: print(type(x)), args)
        #jax.tree_util.tree_map(lambda x: print(x.shape), args)
        lengths = jax.tree_util.tree_multimap(lambda arg, axis: None if axis is None else arg.shape[axis], args, in_axes)
        length = jax.tree_util.tree_leaves(lengths)[0]
        
        nbatchfull = ncpu
        batch_size = length//nbatchfull
        batched_length = nbatchfull*batch_size
        remainder = length%batch_size
        print("batched_length", batched_length)
        
        #args_batched = jax.tree_util.tree_map(lambda x: np.reshape(x[:batched_length], (nbatchfull, batch_size, *x.shape[1:])), args)
        args_batched = jax.tree_util.tree_multimap(lambda x,axis: batch_axis(slice_axis(x,axis,None,batched_length),axis,nbatchfull), args, in_axes)
        #args_batched = jax.tree_util.tree_multimap(lambda x,axis: batch_axis(slice_axis(x,axis,0,batched_length),axis,nbatchfull), args, in_axes)
        #args_batched = jax.tree_util.tree_multimap(lambda x,axis: x if axis is None else np.reshape(x[:batched_length], (nbatchfull, batch_size, *x.shape[1:])), args,in_axes)
        #args_batched = jax.tree_util.tree_multimap(lambda x,axis: x if axis is None else np.reshape(x, (nbatchfull, batch_size, *x.shape[1:])), args,in_axes)
        #args_batched = jax.tree_util.tree_multimap(lambda x,axis: x if axis is None else np.reshape(x, (nbatchfull, x.shape[0]//nbatchfull, *x.shape[1:])), args,in_axes)
        
        #print("args batched shape")
        #for arg in args_batched:
            #print(arg.shape)
        
        if remainder>0:
            #args_remainder = jax.tree_util.tree_map(lambda x: x[batched_length:], args)
            args_remainder = jax.tree_util.tree_multimap(lambda x,axis: slice_axis(x,axis,batched_length,None), args, in_axes)
            #print("args remainder shape")
            #for arg in args_remainder:
                #print(arg.shape)
            #print(args_remainder[0].shape)
            out_remainder = f(*args_remainder)
        else:
            out_remainder = None
        #out_remainder = None
            
        return args_batched, out_remainder
            
    @jax.jit
    def tree_accumulate(x,y):
        return jax.tree_util.tree_multimap(accumulate, x, y)
            
    @jax.jit
    def tree_sum(tree):
        return jax.tree_util.tree_map(lambda x: np.sum(x,axis=0), tree)
            
    def _peval(*args):
        #print("start")
        #if isinstance(in_axes, int):
            #args_flat, args_tree = jax.tree_util.tree_flatten(args)
            #in_axes_flat = (in_axes,)*len(args_flat.shape)
            #in_axes_tree = jax.tree_util.tree_unflatten(args_tree, in_axes_flat)
        #else:
            #in_axes_tree = in_axes
            
        in_axes_tree = in_axes
        
        #args_batched, out_remainder = pbatch(*args, in_axes=in_axes_tree)
        args_batched, out_remainder = pbatch(*args)
        #print("peval")
        out = _pbeval(*args_batched)
        #print("sum")
        out = tree_sum(out)
        
        #print("remainder acc")
        if out_remainder is not None:
            out = tree_accumulate(out, out_remainder)
            
        return out
    
    return _peval

def pbatch_accumulate3(f, batch_size=128, ncpu=32, accumulate=lambda y,x: np.add(y,x)):
    
    @jax.jit
    def _beval(*args):
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]

        nbatchfull = length//batch_size
        batched_length = nbatchfull*batch_size
        remainder = length%batch_size

        args_batched = jax.tree_util.tree_map(lambda x: np.reshape(x[:batched_length], (nbatchfull, batch_size, *x.shape[1:])), args)
        
        args_first = jax.tree_util.tree_map(lambda x: x[0], args_batched)
        out_shape = jax.eval_shape(f,*args_first)
        out_zeros = jax.tree_util.tree_map(np.zeros_like, out_shape)

        def f_scan(y,x):
            fx = f(*x)
            y = jax.tree_util.tree_multimap(accumulate, y, fx)
            return y,()
        #f_scan = jax.jit(f_scan)
        out = jax.lax.scan(f_scan, init=out_zeros, xs=args_batched)[0]

        if remainder>0:
            args_remainder = jax.tree_util.tree_map(lambda x: x[batched_length:], args)
            out_remainder = f(*args_remainder)
            out = jax.tree_util.tree_multimap(accumulate, out, out_remainder)
        return out
    
    _pbeval = jax.pmap(_beval,devices=jax.devices()[:ncpu])
    
    @jax.jit
    def pbatch(*args):
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]
        
        nbatchfull = ncpu
        batch_size = length//nbatchfull
        batched_length = nbatchfull*batch_size
        remainder = length%batch_size
        
        args_batched = jax.tree_util.tree_map(lambda x: np.reshape(x[:batched_length], (nbatchfull, batch_size, *x.shape[1:])), args)
        
        if remainder>0:
            args_remainder = jax.tree_util.tree_map(lambda x: x[batched_length:], args)
            #print(args_remainder[0].shape)
            out_remainder = f(*args_remainder)
        else:
            out_remainder = None
            
        return args_batched, out_remainder
            
    @jax.jit
    def tree_accumulate(x,y):
        return jax.tree_util.tree_multimap(accumulate, x, y)
            
    @jax.jit
    def tree_sum(tree):
        return jax.tree_util.tree_map(lambda x: np.sum(x,axis=0), tree)
            
    def _peval(*args):
        #print("start")
        args_batched, out_remainder = pbatch(*args)
        #print("peval")
        out = _pbeval(*args_batched)
        #print("sum")
        out = tree_sum(out)
        
        #print("remainder acc")
        if out_remainder is not None:
            out = tree_accumulate(out, out_remainder)
            
        return out
    
    return _peval

        
def pbatch_accumulate_simple(f, batch_size=128, ncpu=32, accumulate=lambda y,x: np.add(y,x)):
    
    @jax.jit
    def _beval(arg):
        length = arg.shape[0]

        nbatchfull = length//batch_size
        batched_length = nbatchfull*batch_size
        remainder = length%batch_size

        arg_batched = np.reshape(arg[:batched_length], (nbatchfull, batch_size, *arg.shape[1:]))
        
        arg_first = arg_batched[0]
        out_shape = jax.eval_shape(f,arg_first)
        out_zeros = np.zeros_like(out_shape)

        def f_scan(y,x):
            fx = f(x)
            #y = jax.tree_util.tree_multimap(accumulate, y, fx)
            y = y + fx
            return y,()
        #f_scan = jax.jit(f_scan)
        out = jax.lax.scan(f_scan, init=out_zeros, xs=arg_batched)[0]

        if remainder>0:
            #args_remainder = jax.tree_util.tree_map(lambda x: x[batched_length:], args)
            arg_remainder = arg[batched_length]
            out_remainder = f(arg_remainder)
            out = out + out_remainder
            #out = jax.tree_util.tree_multimap(accumulate, out, out_remainder)
        return out
    
    _pbeval = jax.pmap(_beval,devices=jax.devices()[:ncpu])
    
    @jax.jit
    def pbatch(arg):
        length = arg.shape[0]
        
        nbatchfull = ncpu
        batch_size = length//nbatchfull
        batched_length = nbatchfull*batch_size
        remainder = length%batch_size
        
        arg_batched = np.reshape(arg[:batched_length], (nbatchfull, batch_size, *arg.shape[1:]))
        
        if remainder>0:
            #args_remainder = jax.tree_util.tree_map(lambda x: x[batched_length:], args)
            arg_remainder = arg[batched_length:]
            #print(args_remainder[0].shape)
            out_remainder = f(arg_remainder)
        else:
            out_remainder = None
            
        return arg_batched, out_remainder
            
    @jax.jit
    def tree_accumulate(x,y):
        return jax.tree_util.tree_multimap(accumulate, x, y)
            
    @jax.jit
    def tree_sum(tree):
        return jax.tree_util.tree_map(lambda x: np.sum(x,axis=0), tree)
            
    def _peval(arg):
        #print("start")
        arg_batched, out_remainder = pbatch(arg)
        #print("peval")
        out = _pbeval(arg_batched)
        #print("sum")
        out = np.sum(out,axis=0)
        
        #print("remainder acc")
        if out_remainder is not None:
            out = out + out_remainder
            #out = tree_accumulate(out, out_remainder)
            
        return out
    
    return _peval

        


def pbatch_accumulate2(f, batch_size=4096, ncpu=32, accumulate=lambda y,x: np.add(y,x)):
    #pf = jax.jit(jax.pmap(f))
        
    def sumtrees(y):
        return jax.tree_util.tree_map(lambda x: np.sum(x,axis=0), y)
    sumtrees = jax.jit(sumtrees)
    
    def add_remainder(out, args, batched_length):
        args_remainder = jax.tree_util.tree_map(lambda x: x[batched_length:], args)
        #print(args_remainder[0].shape)
        out_remainder = f(*args_remainder)
        out = jax.tree_util.tree_multimap(accumulate, out, out_remainder)
        return out
    #add_remainder = jax.jit(add_remainder,static_argnums=(2,))
    

    #def get_zeros(args):
        #args_first = jax.tree_util.tree_map(lambda x: x[0,0], args)
        #out_shape = jax.eval_shape(f,*args_first)
        #out_zeros = jax.tree_util.tree_map(np.zeros_like, out_shape)
        #return out_zeros
    #get_zeros = jax.jit(get_zeros)
    
    def _seval(args):
        def get_zeros(args):
            args_first = jax.tree_util.tree_map(lambda x: x[0], args)
            out_shape = jax.eval_shape(f,*args_first)
            out_zeros = jax.tree_util.tree_map(np.zeros_like, out_shape)
            return out_zeros
        
        def f_scan(y,x):
                #print(x[0].shape)
                fx = f(*x)
                #pfx = pf(*x)
                #fx = jax.tree_util.tree_map(lambda x: np.sum(x,axis=0), pfx)
                y = jax.tree_util.tree_multimap(accumulate, y, fx)
                return y,()

        out_zeros = get_zeros(args)
        #f_scan = jax.jit(f_scan)
        out = jax.lax.scan(f_scan, init=out_zeros, xs=args)[0]
        return out
    
    peval = jax.pmap(jax.jit(_seval))
    
    def _beval(*args):
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]

        nbatchfull = length//batch_size
        batched_length = nbatchfull*batch_size
        remainder = length%batch_size
        
        #print("nbatchfull", nbatchfull)
        #print("batched_length", batched_length)
        #print("remainder", remainder)
        

        #args_batched = jax.tree_util.tree_map(lambda x: np.reshape(x[:batched_length], (nbatchfull, ncpu, batch_size//ncpu, *x.shape[1:])), args)
        args_batched = jax.tree_util.tree_map(lambda x: np.reshape(x[:batched_length], (ncpu, nbatchfull, batch_size//ncpu, *x.shape[1:])), args)
        
        print("get shape")

        #out_zeros = get_zeros(args_batched)


        print("pmap")
        #pout = jax.pmap(jax.jit(_seval))(args_batched)
        pout = peval(args_batched)
        print("sum")
        #jax.tree_util.tree_map(lambda x: print(x.shape), pout)

        out = sumtrees(pout)
        #out = jax.tree_util.tree_map(lambda x: np.sum(x,axis=0), pout)
        #out = jax.tree_util.tree_map(lambda x: x[0], pout)
        #out = jax.pmap(lambda x: jax.lax.psum(_seval(x),'i'), axis_name='i')(args_batched)



        print("remainder")
        #remainder=0
        if remainder>0:
            out = add_remainder(out,args, batched_length)

            #print("remainder")
            ##remainder=0
            #if remainder>0:
                #args_remainder = jax.tree_util.tree_map(lambda x: x[batched_length:], args)
                ##print(args_remainder[0].shape)
                #out_remainder = f(*args_remainder)
                #out = jax.tree_util.tree_multimap(accumulate, out, out_remainder)
        print("done")
        return out
    return _beval

def lbatch_accumulate(f, batch_size=128, in_axes=0, accumulate=lambda y,x: np.add(y,x)):
    
    def slice_axis(x, axis, start, end):
        if axis is None:
            return x
        else:
            slc = [slice(None)] * len(x.shape)
            slc[axis] = slice(start,end)
            return x[tuple(slc)]
    
    def _beval(*args):
        lengths = jax.tree_util.tree_multimap(lambda arg, axis: None if axis is None else arg.shape[axis], args, in_axes)
        length = jax.tree_util.tree_leaves(lengths)[0]
        
        idxstart = np.arange(0,length,batch_size)

        outval = None
        
        
        for idx in idxstart:
            if isinstance(in_axes,int):
                args_batch = jax.tree_util.tree_map(lambda x: slice_axis(x, in_axes, idx, idx+batch_size), args)
            else:
                args_batch = jax.tree_util.tree_multimap(lambda x,axis: slice_axis(x,axis,idx,idx+batch_size), args, in_axes)
            val = f(*args_batch)
            if outval is None:
                outval = val
            else:
                outval = jax.tree_util.tree_multimap(accumulate, outval, val)
                
        return outval
    return _beval

def lbatch(f, batch_size=128, in_axes=0):
    
    def slice_axis(x, axis, start, end):
        if axis is None:
            return x
        else:
            slc = [slice(None)] * len(x.shape)
            slc[axis] = slice(start,end)
            return x[tuple(slc)]
    
    def _beval(*args):
        lengths = jax.tree_util.tree_multimap(lambda arg, axis: None if axis is None else arg.shape[axis], args, in_axes)
        length = jax.tree_util.tree_leaves(lengths)[0]
        
        idxstart = np.arange(0,length,batch_size)

        outval = None
        
        for idx in idxstart:
            if isinstance(in_axes,int):
                args_batch = jax.tree_util.tree_map(lambda x: slice_axis(x, in_axes, idx, idx+batch_size), args)
            else:
                args_batch = jax.tree_util.tree_multimap(lambda x,axis: slice_axis(x,axis,idx,idx+batch_size), args, in_axes)
            val = f(*args_batch)
            if outval is None:
                outval = val
            else:
                outval = jax.tree_util.tree_multimap(lambda x,y: np.concatenate((x,y),axis=0), outval, val)
                
        return outval
    return _beval

def random_subset(f, subset, in_axes=0, rescale=True):
    def _eval(*args):
        lengths = jax.tree_util.tree_multimap(lambda arg, axis: None if axis is None else arg.shape[axis], args, in_axes)
        length = jax.tree_util.tree_leaves(lengths)[0]
        
        idxs = onp.random.choice(length, np.minimum(subset,length))
        
        args_subset = jax.tree_util.tree_multimap(lambda x,axis: x if axis is None else x[idxs], args, in_axes)
        
        val = f(*args_subset)
        if rescale:
            val = jax.tree_util.tree_map(lambda x: (length/subset)*x, val)
        
        return val
        
        #return f(*args_subset)
    return _eval


def lpbatch_accumulate(f, batch_size=128,ncpu=32, in_axes=0, accumulate=lambda y,x: np.add(y,x)):
    
    
    def slice_axis(x, axis, start, end):
        if axis is None:
            return x
        else:
            slc = [slice(None)] * len(x.shape)
            slc[axis] = slice(start,end)
            return x[tuple(slc)]
    
    print("in_axes", in_axes)
    #pf = jax.pmap(f, None, in_axes)
    #pf = jax.pmap(f, None, 0)
    #pf = jax.pmap(f,0,0)
    pf = jax.pmap(f, in_axes=in_axes, devices=jax.devices()[:ncpu])
    #pf = jax.pmap(f, in_axes=in_axes)
    #pf = jax.jit(pf)
    
    def _beval(*args):
        pbatch_size = batch_size*ncpu
        
        #TODO, construct proper in_axes tree from int case
        
        lengths = jax.tree_util.tree_multimap(lambda arg, axis: None if axis is None else arg.shape[axis], args, in_axes)
        length = jax.tree_util.tree_leaves(lengths)[0]
        
        length = args[1].shape[0]

        idxstart = np.arange(0,length,pbatch_size)

        outval = None
        
        #print("tree_map type check:")
        #jax.tree_util.tree_map(lambda x: print(type(x)), args)
        
        print("starting loop")
        for idx in idxstart:
            #if isinstance(in_axes,int):
                #args_batch = jax.tree_util.tree_map(lambda x: slice_axis(x, in_axes, idx, idx+batch_size), args)
            #else:
            args_batch = jax.tree_util.tree_multimap(lambda x,axis: slice_axis(x,axis,idx,idx+pbatch_size), args, in_axes)
            size = min(idx+pbatch_size, length) - idx
            
            #print("arg_batch shape")
            #for arg in args_batch:
                #print(arg.shape)
            
            if size%ncpu==0:
                args_batch = jax.tree_util.tree_multimap(lambda x,axis: x if axis is None else np.reshape(x, (ncpu,size//ncpu)+x.shape[1:]), args_batch, in_axes)
                val = pf(*args_batch)
                val = jax.tree_util.tree_map(lambda x: np.sum(x,axis=0), val)
            else:
                print("remainder")
                #val = f(*args_batch)
                val = None
                jstep = size//ncpu*ncpu
                if jstep>0:
                    jdxstart = np.arange(0,size,jstep)
                else:
                    jdxstart = [0]
                    jstep = size
                for jdx in jdxstart:
                    print("subremainder")
                    args_subbatch = jax.tree_util.tree_multimap(lambda x,axis: slice_axis(x,axis,jdx,jdx+jstep), args_batch, in_axes)
                    #print("arg subbatch shape")
                    #for arg in args_subbatch:
                        #print(arg.shape)
                    jsize = min(jdx+jstep, size) - jdx
                    if jsize%ncpu==0:
                        args_subbatch = jax.tree_util.tree_multimap(lambda x,axis: x if axis is None else np.reshape(x, (ncpu,jsize//ncpu)+x.shape[1:]), args_subbatch, in_axes)
                        subval = pf(*args_subbatch)
                        subval = jax.tree_util.tree_map(lambda x: np.sum(x,axis=0), subval)
                    else:
                        subval = f(*args_subbatch)
                    if val is None:
                        val = subval
                    else:
                        val = jax.tree_util.tree_multimap(accumulate, val, subval)
                        
            if outval is None:
                outval = val
            else:
                outval = jax.tree_util.tree_multimap(accumulate, outval, val)
                
        return outval
    return _beval


def lpbatch_accumulate6(f, batch_size=128, ncpu=32, accumulate=lambda y,x: np.add(y,x)):
    #f = jax.jit(f)
    def _beval(*args):
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]

        idxstart = np.arange(0,length,batch_size)

        args_first = jax.tree_util.tree_map(lambda x: x[:batch_size], args)
        out_shape = jax.eval_shape(f, *args_first)
        out_zeros = jax.tree_util.tree_map(np.zeros_like, out_shape)
        
        def f_scan(y,x):
            idx = x
            #arg_scan = jax.tree_util.tree_multimap(lambda x: x[idx:idx+batch_size], args)
            arg_scan = jax.tree_util.tree_multimap(lambda x: jax.lax.dynamic_slice_in_dim(x,idx,batch_size,axis=0), args)
            fx = f(*arg_scan)
            y = jax.tree_util.tree_multimap(accumulate, y, fx)
            return y,()

        out = jax.lax.scan(f_scan, init=out_zeros, xs=idxstart)[0]
        return out
    
    #bevals = [jax.jit(_beval,device=jax.devices()[icpu]) for icpu in range(ncpu)]
        
    #def seval(idx,step,*args):
        ##arg_p = jax.tree_util.tree_map(lambda x: x[idx:idx+step], args)
        #arg_p = jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x,idx,step,axis=0), args)
        #return _beval(*arg_p)
    
    #def seval2(idx):
        #return 
    
    
    #psevel = lambda idx,step,*args: jax.pmap(
    #pseval = jax.pmap(seval, static_broadcasted_argnums=(1,2,))
        
    def _peval(*args):
        print("start")
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]

        #TODO, this will not work properly in python 2 because of integer division behaviour
        step = np.ceil(length/ncpu).astype(np.int32)
        idxstart = np.arange(0,length,step)
        
        def seval(idx):
            #arg_p = jax.tree_util.tree_map(lambda x: x[idx:idx+step], args)
            arg_p = jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x,idx,step,axis=0), args)
            return _beval(*arg_p)
        
        print("eval")
        outval = jax.pmap(seval,devices=jax.devices()[:ncpu])(idxstart)
        #outval = pseval(idxstart,step,*args)
        
        print("sum")
        outval = jax.tree_util.tree_map(lambda x: np.sum(x,axis=0), outval)
        #outval = jax.tree_util.tree_multimap(lambda *args: np.sum(np.stack(args, axis=0),axis=0), *outvals)
        print("done")
        return outval
        
    
    return _peval

def lpbatch_accumulate4(f, batch_size=128, ncpu=32, accumulate=lambda y,x: y+x):
    #f = jax.jit(f)
    def _beval(f, *args):
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]

        idxstart = np.arange(0,length,batch_size)

        outval = None
        
        for idx in idxstart:
            args_batch = jax.tree_util.tree_map(lambda x: x[idx:idx+batch_size], args)
            val = f(*args_batch)
            if outval is None:
                outval = val
            else:
                outval = jax.tree_util.tree_multimap(accumulate, outval, val)
                
        return outval
        
    bevals = [lambda *args: _beval(jax.jit(f,device=jax.devices()[icpu]),*args) for icpu in range(ncpu)]

        
    def _peval(*args):
        print("start")
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]

        #TODO, this will not work properly in python 2 because of integer division behaviour
        step = np.ceil(length/ncpu).astype(np.int32)
        idxstart = np.arange(0,length,step)
        
        print("split")
        args_p = []
        for idx in idxstart:
            #icpu = i%ncpu
            arg_p = jax.tree_util.tree_map(lambda x: x[idx:idx+step], args)
            args_p.append(arg_p)
            #outvals.append(jax.jit(_beval, device=jax.devices()[icpu])(*arg_p))
            #outvals.append(bevals[icpu](*arg_p))
            
        print("eval")
        outvals = []
        for i,arg_p in enumerate(args_p):
            
            icpu = i%ncpu
            #outval = jax.jit(_beval, device=jax.devices()[icpu])(*arg_p)
            #outval = bevals[icpu](*arg_p)
            #print(outval)
            #outval = _beval(jax.jit(f,device=jax.devices()[icpu]), *arg_p)
            print("eval i", i)
            outval = bevals[icpu](*arg_p)
            print("done eval i", i)
            outvals.append(outval)
            print("done append i", i)
            
        print("sum")
        outval = jax.tree_util.tree_multimap(lambda *args: np.sum(np.stack(args, axis=0),axis=0), *outvals)
        print("done")
        return outval
    
    return _peval

def lpbatch_accumulate5(f, batch_size=128, ncpu=32, accumulate=lambda y,x: np.add(y,x)):
    #f = jax.jit(f)
    def _beval(*args):
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]

        idxstart = np.arange(0,length,batch_size)

        args_first = jax.tree_util.tree_map(lambda x: x[:batch_size], args)
        out_shape = jax.eval_shape(f, *args_first)
        out_zeros = jax.tree_util.tree_map(np.zeros_like, out_shape)
        
        def f_scan(y,x):
            idx = x
            #arg_scan = jax.tree_util.tree_multimap(lambda x: x[idx:idx+batch_size], args)
            arg_scan = jax.tree_util.tree_multimap(lambda x: jax.lax.dynamic_slice_in_dim(x,idx,batch_size,axis=0), args)
            fx = f(*arg_scan)
            y = jax.tree_util.tree_multimap(accumulate, y, fx)
            return y,()

        out = jax.lax.scan(f_scan, init=out_zeros, xs=idxstart)[0]
        return out
    
    bevals = [jax.jit(_beval,device=jax.devices()[icpu]) for icpu in range(ncpu)]
        
    def _peval(*args):
        print("start")
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]

        #TODO, this will not work properly in python 2 because of integer division behaviour
        step = np.ceil(length/ncpu).astype(np.int32)
        idxstart = np.arange(0,length,step)
        
        print("split")
        args_p = []
        for idx in idxstart:
            #icpu = i%ncpu
            arg_p = jax.tree_util.tree_map(lambda x: x[idx:idx+step], args)
            args_p.append(arg_p)
            #outvals.append(jax.jit(_beval, device=jax.devices()[icpu])(*arg_p))
            #outvals.append(bevals[icpu](*arg_p))
            
        print("eval")
        outvals = []
        for i,arg_p in enumerate(args_p):
            #print("eval i", i)
            icpu = i%ncpu
            #outval = jax.jit(_beval, device=jax.devices()[icpu])(*arg_p)
            print("eval i", i)
            outval = bevals[icpu](*arg_p)
            print("done eval i", i)
            #print(outval)
            #outval = _beval(jax.jit(f,device=jax.devices()[icpu]), *arg_p)
            outvals.append(outval)
            print("done append i", i)
            
        print("sum")
        outval = jax.tree_util.tree_multimap(lambda *args: np.sum(np.stack(args, axis=0),axis=0), *outvals)
        print("done")
        return outval
    
    return _peval

def lpbatch_accumulate3(f, batch_size=128, ncpu=32, accumulate=lambda y,x: np.add(y,x)):
    f = jax.jit(f)
    def facc(y, x):
        return jax.tree_util.tree_multimap(accumulate, y,f(*x))
    facc = jax.jit(facc)
    def _beval(*args):
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]

        idxstart = np.arange(0,length,batch_size)

        outvals = ncpu*[None]
        
        for i,idx in enumerate(idxstart):
            icpu = i%ncpu
            args_batch = jax.tree_util.tree_map(lambda x: x[idx:idx+batch_size], args)
            #val = jax.jit(f,device=jax.devices()[icpu])(*args_batch)
            if outvals[icpu] is None:
                outvals[icpu] = jax.jit(f,device=jax.devices()[icpu])(*args_batch)
            else:
                outvals[icpu] = jax.jit(facc,device=jax.devices()[icpu])(outvals[icpu], args_batch)
                #outvals[icpu] = jax.tree_util.tree_multimap(accumulate, outvals[icpu], val)
                
        #return outval
        outval = jax.tree_util.tree_multimap(lambda *args: np.sum(np.stack(args, axis=0),axis=0), *outvals)
        return outval
    return _beval

def lpbatch_accumulate2(f, batch_size=4096, ncpu=32, accumulate=lambda y,x: np.add(y,x)):
    pf = jax.jit(jax.pmap(f,devices=jax.devices()))
    #pf = jax.jit(jax.vmap(f))
    def _beval(*args):
        args_flat, args_tree = jax.tree_util.tree_flatten(args)
        length = args_flat[0].shape[0]

        idxstart = np.arange(0,length,batch_size)

        outval = None
        
        for idx in idxstart:
            args_batch = jax.tree_util.tree_map(lambda x: x[idx:idx+batch_size], args)
            if idx+batch_size <= length:
                 args_pmapped = jax.tree_util.tree_map(lambda x: np.reshape(x, (ncpu, batch_size//ncpu, *x.shape[1:])), args_batch)
                 val_pmapped = pf(*args_pmapped)
                 val = jax.tree_util.tree_map(lambda x: np.sum(x,axis=0), val_pmapped)
            else:
                val = f(*args_batch)
            if outval is None:
                outval = val
            else:
                outval = jax.tree_util.tree_multimap(accumulate, outval, val)
                
        return outval
    return _beval




def vgrad(f, batch_size=128):
    #vg = jax.vmap(jax.jacfwd(f, holomorphic=True))
    vg = jax.vmap(jax.grad(f))
    vg = jax.jit(vg)
        
    def _vgrad(x, *args):
        n = x.shape[0]
        idxstart = np.arange(0,n,batch_size)
        idxend =  np.arange(batch_size, n, batch_size)
        if idxend.shape[0]<idxstart.shape[0]:
            idxend = np.concatenate((idxend,np.array([n])))

        outs = []
        for istart,iend in zip(idxstart,idxend):
            argsplit = []
            for arg in args:
                argsplit.append(arg[istart:iend])
            outs.append(vg(x[istart:iend],*argsplit))
            
        return np.concatenate(outs,axis=0)
    return _vgrad

def hgrad(f, batch_size=128):
    #vg = jax.vmap(jax.jacrev(jax.grad(f,holomorphic=True),holomorphic=True))
    vg = jax.vmap(jax.jacfwd((jax.jacrev(f))))
    #vg = jax.vmap(jax.hessian(f, holomorphic=True))
    vg = jax.jit(vg)

    
    def _vgrad(x, *args):
        n = x.shape[0]
        idxstart = np.arange(0,n,batch_size)
        idxend =  np.arange(batch_size, n, batch_size)
        if idxend.shape[0]<idxstart.shape[0]:
            idxend = np.concatenate((idxend,np.array([n])))

        outs = []
        for istart,iend in zip(idxstart,idxend):
            print("iter")
            argsplit = []
            for arg in args:
                argsplit.append(arg[istart:iend])
            outs.append(vg(x[istart:iend],*argsplit))
            
            
        return np.concatenate(outs,axis=0)
    return _vgrad
        

#def hgrad(f, batch_size=128):
    ##vg = jax.vmap(jax.jacrev(jax.grad(f,holomorphic=True),holomorphic=True))
    #vg = jax.vmap(jax.jacfwd((jax.jacrev(f))))
    ##vg = jax.vmap(jax.hessian(f, holomorphic=True))
    ##vg = jax.jit(vg)
    
    #vgs = []
    #for i in range(32):
        #vgs.append(jax.jit(vg, device=jax.devices()[i]))

    
    #def _vgrad(x, *args):
        #n = x.shape[0]
        #idxstart = np.arange(0,n,batch_size)
        #idxend =  np.arange(batch_size, n, batch_size)
        #if idxend.shape[0]<idxstart.shape[0]:
            #idxend = np.concatenate((idxend,np.array([n])))

        ##outs = []
        ##for istart,iend in zip(idxstart,idxend):
            ##print("iter")
            ##argsplit = []
            ##for arg in args:
                ##argsplit.append(arg[istart:iend])
            ##outs.append(vg(x[istart:iend],*argsplit))
            
        #lx = []
        #largs = []
        #for istart,iend in zip(idxstart,idxend):
            #print("iter")
            #argsplit = []
            #for arg in args:
                #argsplit.append(arg[istart:iend])
            #largs.append(argsplit)
            #lx.append(x[istart:iend])
            ##outs.append(vg(x[istart:iend],*argsplit))
        
        #outs = []
        #for i,(ix,iargs) in enumerate(zip(lx,largs)):
            #print("iter2")
            #ivg = vgs[i%32]
            #outs.append(ivg(ix,*iargs))
            
        #return np.concatenate(outs,axis=0)
    #return _vgrad
        
        
def hpgrad(f):
    vg = jax.vmap(jax.jacfwd((jax.jacrev(f))))
    vg = jax.jit(jax.pmap(vg))
    def _vgrad(x,*args):
        ncpu = 2
        n = x.shape[0]
        npadded = np.ceil(n/ncpu).astype(np.int32)*ncpu
        npad = npadded-n
        print(n, npadded, npad)
        xpad = np.pad(x,[(0,npad),(0,0)])
        xpad = np.reshape(xpad,(ncpu,-1,*x.shape[1:]))
        argspad = []
        for arg in args:
            print(arg.shape)
            argpad = np.pad(arg,[(0,npad),(0,0)])
            print(argpad.shape)
            argpad = np.reshape(argpad,(ncpu,-1,*arg.shape[1:]))
            argspad.append(argpad)
        res = jax.pmap(vg(xpad,*argspad))
        res = res = np.reshape(res,(-1,*res.shape[2:]))
        res = res[:n]
        return res
    return _vgrad
            
        
def flatten(x, val_shape=()):
    #flatten tree
    x_flat, x_tree = jax.tree_util.tree_flatten(x)
    
    #keep track of shapes
    shapes = [arr.shape for arr in x_flat]
    
    #flatten each array (within each parallel minimization if applicable)
    x_flat_arrs = [np.reshape(arr, val_shape + (-1,)) for arr in x_flat]
    
    #keep track of lengths along flattened axis
    split_idx = 0
    split_idxs = []
    for arr in x_flat_arrs:
        split_idx += arr.shape[-1]
        split_idxs.append(split_idx)
    
    #concatenate flat arrays
    x_flat_arr = np.concatenate(x_flat_arrs, axis=-1)
    
    return x_flat_arr, split_idxs, shapes, x_tree

def unflatten(x_tree, shapes, split_idxs, x_flat_arr):
    #split array
    x_flat_arrs = np.split(x_flat_arr, split_idxs, axis=-1)
    #restore original shapes
    x_flat = [np.reshape(arr, shape) for arr,shape in zip(x_flat_arrs, shapes)]
    #unflatten tree
    x = jax.tree_util.tree_unflatten(x_tree, x_flat)
    return x

def flatten_hessian(x, grad_shapes, val_shape=()):
    ngrad = len(grad_shapes)
    x_flat, _ = jax.tree_util.tree_flatten(x)
    
    val_dim = len(val_shape)
    grad_shapes_flat = [ (np.prod(np.array(shape[val_dim:])),) for shape in grad_shapes]
    
    x_square = [x_flat[i*ngrad:(i+1)*ngrad] for i in range(ngrad)]
    
    x_flatarr = []
    for i in range(ngrad):
        for j in range(ngrad):
            flatarr = np.reshape(x_square[i][j], val_shape + grad_shapes_flat[i] + grad_shapes_flat[j])
            x_flatarr.append(flatarr)
            
    x_square_flatarr = [x_flatarr[i*ngrad:(i+1)*ngrad] for i in range(ngrad)]
    
    return np.block(x_square_flatarr)
            


def pmin(f, x, args = [], doParallel=True, jac=False, h = None, lb = None, ub=None, xtol = 0., edmtol = np.sqrt(np.finfo('float64').eps), reqposdef=True, tr_init=1., maxiter = int(100e3)):
    
    #FIXME remove or handle properly for nonflat x
    #if lb is None:
        #lb = -np.inf*np.ones_like(x)
    #if ub is None:
        #ub = np.inf*np.ones_like(x)
    
    ##convert to flat function
    #if doParallel:
        #out_shape = jax.eval_shape(jax.vmap(f),x,*args)[1:]
    #else:
        #out_shape = jax.eval_shape(f,x,*args)
    
    #x,split_idxs,shapes,x_tree = jax.vmap(flatten)(x)
    
    #print("flatten info")
    #print(split_idxs, shapes)
    
    #def f_flat(x_flat, *args):
        #print("x_flat.shape", x_flat.shape)
        #xin = unflatten(x_tree, shapes, split_idxs, x_flat)
        #return f(xin, *args)
        
    #tol = np.sqrt(np.finfo('float64').eps)
    #maxiter = int(100e3)
    

    
    #f = jax.jit(f)
    

    if not jac:
        batch_size_grad = int(2048)
        batch_size = int(512)
        if doParallel:
            #g = jax.grad(lambda x,*args: np.sum(f(x,*args),axis=0))
            #g = jax.jit(g)
            
            #fg = jax.jit(jax.vmap(jax.value_and_grad(f)))
            #fg = beval(fg, accumulator=lambda x: np.concatenate(x,axis=0), batch_size=512)
            #fg = jax.jit(fg)
            
            #h  = jax.jit(jax.vmap(jax.hessian(f)))
            #h = beval(h, accumulator=lambda x: np.concatenate(x,axis=0), batch_size=512)
            #h = jax.jit(h)
            
            fg = jax.jit(batch_vmap(jax.value_and_grad(f), batch_size=256))
            h = jax.jit(batch_vmap(jax.hessian(f), batch_size=256))
            
            #hi = jax.jacrev(lambda x,*args: np.sum(g(x,*args),axis=0))
            #h = lambda x,*args: np.swapaxes(hi(x,*args),0,1)
            #h = jax.jit(h)
            
            #g = jax.jit(jax.vmap(jax.grad(f)))
            #h = jax.vmap(jax.hessian(f))
            #h = jax.jit(h,backend="cpu")
            
            #h = jax.jit(jax.vmap(jax.hessian(f)))
            #h = jax.
            #g = vgrad(f,batch_size=batch_size_grad)
            #h = hgrad(f,batch_size=batch_size)
            #h = hpgrad(f)
            #h = jax.jit(jax.hessian(f))
        else:
            fg = jax.jit(jax.value_and_grad(f))
            h = jax.jit(jax.hessian(f))
        #h = None
    else:
        fg = f
        
    
    
    print("starting fit")
    
    #fiter = jax.jit(piter, static_argnums=(0,4))
    
    
    val, grad = fg(x,*args)
    
    x_flat, split_idxs, shapes, x_tree = flatten(x, val.shape)
    hess_flat = np.eye(x_flat.shape[-1], dtype=np.float64)
    hess_flat = hess_flat*np.ones(shape=x_flat.shape[:-1] + 2*(x_flat.shape[-1],), dtype=np.float64)
    print("hess_flat shape", hess_flat.shape)
    
    trust_radius = tr_init*np.ones(shape=x_flat.shape[:-1], dtype=x_flat.dtype)
    
    #@jax.jit
    #def piter_(x, val, grad, hess_flat, trust_radius):
        #return piter(fg,h,x,val,grad,hess_flat,trust_radius,args,lb,ub)


    for i in range(maxiter):
        x, val, grad, hess_flat, trust_radius, edm, e0 = piter(fg,h,x,val,grad,hess_flat,trust_radius,args,lb,ub)
        #x, val, grad, hess_flat, trust_radius, edm, e0 = piter_(x,val,grad,hess_flat,trust_radius)
        #FIXME remove or make this work with nonflat x
        #gradmag = np.linalg.norm(grad,axis=-1)
        gradmag = 0
        print("iter", i, np.sum(val), np.max(trust_radius), np.max(gradmag), np.sum(edm), np.max(edm), np.min(e0))
        print(x)
        if reqposdef:
            convergence = np.logical_and(edm<edmtol, e0>0.)
        else:
            convergence = np.abs(edm)<edmtol
        convergence = np.logical_or(convergence, trust_radius<xtol)
        convergence = np.all(convergence)
        if convergence:
            break        
    
    #x = unflatten(x_tree, shapes, split_idxs, x)
    return x

def piter(fg,h,x,val,grad,hess_flat,trust_radius, args, lb, ub):
        #flatten and get shape info
        x_flat, split_idxs, shapes, x_tree = flatten(x, val.shape)
        grad_flat = flatten(grad, val.shape)[0]
        print("grad_flat.shape", grad_flat.shape)

        #lb_flat = flatten(lb, val.shape)[0]
        #ub_flat = flatten(ub, val.shape)[0]
        
        #FIXME remove or handle properly for nonflat x
        lb_flat = -np.inf*np.ones_like(x_flat)
        ub_flat = np.inf*np.ones_like(x_flat)

        if h is not None:
            hess = h(x,*args)
            #hess = np.eye(x.shape[-1],dtype=x.dtype)
            #hess = np.expand_dims(hess,axis=0)
            print("done evaluating")
            
            #flatten hessian
            hess_flat = flatten_hessian(hess,shapes,val.shape)
        
        print("eigendecomposition")
        #e,u = eigh(hess)
        #print(hess_flat)
        e,u = eigh(hess_flat)
        #e.block_until_ready()
        print("eigendecomposition done")
        #print("hess_flat error check")
        #print(np.any(np.isnan(hess_flat) | np.isinf(hess_flat), axis=(-1,-2)))
        #print("eigenvalues error check")
        #print(np.any(np.isnan(e) | np.isinf(e), axis=-1))
        
        e0 = e[...,0]
        print("next")

        print("tr_solve")
        p, at_boundary, predicted_reduction, edm = tr_solve(grad_flat,e,u,trust_radius)
        #p, at_boundary, predicted_reduction, edm = tr_solve(grad,e,u,trust_radius)
        print("tr_solve done")


        print("x_flat shape", x_flat.shape)
        print("p.shape", p.shape)
        #compute actual reduction in loss
        x_new_flat = x_flat + p
        #unflatten to original shapes and tree structure
        #print(x_new_flat.shape, shapes, split_idxs)
        print("shapes", shapes)
        x_new = unflatten(x_tree, shapes, split_idxs, x_new_flat)
        val_new, grad_new = fg(x_new, *args)
        grad_new_flat = flatten(grad_new, val.shape)[0]
        
        #Roofit style "return large number to minimizer" protection which should reject solutions resulting in NaN
        #and reduce the trust region size
        invalid = np.isinf(val_new) | np.isnan(val_new) | np.any(np.isnan(grad_new_flat) | np.isinf(grad_new_flat), axis=-1)
        #invalid = np.isnan(val_new) | np.any(np.isnan(grad_new_flat), axis=-1)
        #invalid = invalid & np.any(x_new_flat<lb_flat | x_new_flat>ub_flat, axis=-1)
        invalid = np.logical_or(invalid, np.any(np.logical_or(x_new_flat<lb_flat, x_new_flat>ub_flat), axis=-1))
        val_new = np.where(invalid, np.inf, val_new)
        grad_new_flat = np.where(invalid[...,np.newaxis], grad_flat, grad_new_flat)
        
        actual_reduction = val - val_new
        
        #update trust radius and output parameters, following Nocedal and Wright 2nd ed. Algorithm 4.1 or 6.2 as appropriate
        rho = actual_reduction/np.where(np.equal(actual_reduction,0.), 1., predicted_reduction)
        if h is not None:
            eta = 0.15
            #trust_radius_max = 1e3
            trust_radius_max = 1e16
            trust_radius_out = np.where(rho<0.25, 0.25*trust_radius, np.where(np.logical_and(rho>0.75,at_boundary),np.minimum(2.*trust_radius, trust_radius_max),trust_radius))
        else:
            eta = 1e-3
            pmag = np.linalg.norm(p,axis=-1)
            #trust_radius_max = 1e3
            trust_radius_max = 1e16
            trust_radius_out = np.where(rho>0.75, np.where(pmag<=0.8*trust_radius, trust_radius, np.minimum(2.*trust_radius, trust_radius_max)), np.where(np.logical_and(rho>=0.1, rho<=0.75), trust_radius, 0.5*trust_radius))
        
        acceptsol = rho>eta
        #compute hessian only if needed
        #hess_new = jax.lax.cond(np.any(acceptsol), None, lambda _: h(x_new,*args), None, lambda _: hess)
        x_out_flat = np.where(acceptsol[...,np.newaxis], x_new_flat, x_flat)
        val_out = np.where(acceptsol, val_new, val)
        grad_out_flat = np.where(acceptsol[...,np.newaxis], grad_new_flat, grad_flat)
        #hess_out = np.where(acceptsol[...,np.newaxis,np.newaxis], hess_new, hess)
        
        if h is None:
            hess_new_flat = sr1Update(hess_flat, grad_new_flat-grad_flat, x_new_flat-x_flat)
            hessinvalid = np.any(np.isnan(hess_new_flat) | np.isinf(hess_new_flat), axis=(-1,-2), keepdims=True) | invalid[...,np.newaxis,np.newaxis]
            #hess_out_flat = np.where(invalid[...,np.newaxis,np.newaxis], hess_flat, hess_new_flat)
            hess_out_flat = np.where(hessinvalid, hess_flat, hess_new_flat)
        else:
            hess_out_flat = hess_flat
            
        print("hess_flat. shape", hess_flat.shape)
        print("hess_out_flat.shape", hess_out_flat.shape)
        
        x_out = unflatten(x_tree, shapes, split_idxs, x_out_flat)
        grad_out = unflatten(x_tree, shapes, split_idxs, grad_out_flat)
        #print("grad_out_flat:")
        #print(grad_out_flat)
        
        return x_out, val_out, grad_out, hess_out_flat, trust_radius_out, edm, e0

eigh = jax.jit(np.linalg.eigh)

@jax.jit
def tr_solve(grad, e, u, trust_radius):

        #compute function value and gradient
        #val,grad = fg(x,*args)
        gradcol = np.expand_dims(grad, axis=-1)
        
        #compute hessian and eigen-decomposition
        #hess = h(x,*args)
        #hess = np.eye(x.shape[0],dtype=x.dtype)
        #e,u = np.linalg.eigh(hess)
        
        ut = np.swapaxes(u,-1,-2)
        
        #convert gradient to eigen-basis
        a = np.matmul(ut,gradcol)
        a = np.squeeze(a, axis=-1)
        
        lam = e
        e0 = lam[...,0]
        
        #compute estimated distance to minimum for unconstrained solution (only valid if e0>0)
        coeffs0 = -a/lam
        edm = -np.sum(a*coeffs0 + 0.5*lam*np.square(coeffs0), axis=-1)
        
        #TODO deal with null gradient components and repeated eigenvectors
        lambar = lam
        abarsq = np.square(a)

        
        def phif(s):
            s = np.expand_dims(s,-1)
            pmagsq = np.sum(abarsq/np.square(lambar+s),axis=-1)
            pmag = np.sqrt(pmagsq)
            phipartial = np.reciprocal(pmag)
            singular = np.any(np.equal(-s,lambar),axis=-1)
            phipartial = np.where(singular, 0., phipartial)
            phi = phipartial - np.reciprocal(trust_radius)
            return phi
        
        def phiphiprime(s):
            phi = phif(s)
            s = np.expand_dims(s,-1)
            pmagsq = np.sum(abarsq/np.square(lambar+s),axis=-1)        
            phiprime = np.power(pmagsq,-1.5)*np.sum(abarsq/np.power(lambar+s,3),axis=-1)
            return (phi, phiprime)
        
        #check if unconstrained solution is valid
        sigma0 = np.maximum(-e0, 0.)
        phisigma0 = phif(sigma0)
        usesolu = np.logical_and(e0>0., phisigma0>=0.)
        
        sigma = np.max(np.abs(a)/trust_radius[...,np.newaxis] - lam, axis=-1)
        sigma = np.maximum(sigma, 0.)
        sigma = np.where(usesolu, 0., sigma)
        phi, phiprime = phiphiprime(sigma)
        
        
        #TODO, add handling of additional cases here (singular and "hard" cases)
        
        #iteratively solve for sigma, enforcing unconstrained solution sigma=0 where appropriate
        unconverged = np.ones(shape=sigma.shape, dtype=np.bool_)
        j = 0
        maxiter=1000

        
        #This can't work with vmap+jit because of the dynamic condition, so we use the jax while_loop below
        #j = 0
        #while np.logical_and(np.any(unconverged), j<maxiter):
            #sigma = sigma - phi/phiprime
            #sigma = np.where(usesolu, 0., sigma)
            #phiout, phiprimeout = phiphiprime(sigma)
            #unconverged = np.logical_and( (phiout > phi) , (phiout < 0.) )
            #phi,phiprime = (phiout, phiprimeout)
            #j = j +1

        def cond(vals):
            sigma, phi, phiprime, unconverged, j = vals
            return np.logical_and(np.any(unconverged) , j<maxiter)

        def body(vals):
            sigma, phi, phiprime, unconverged, j = vals
            sigma = sigma - phi/phiprime
            sigma = np.where(usesolu, 0., sigma)
            phiout, phiprimeout = phiphiprime(sigma)
            unconverged = np.logical_and( (phiout > phi) , (phiout < 0.) )
            phi,phiprime = (phiout, phiprimeout)
            j = j +1
            return (sigma, phi, phiprime, unconverged, j)
        
        sigma = jax.lax.while_loop(cond, body, (sigma,phi,phiprime,unconverged,j))[0]
        
            
        #compute solution from eigenvalues and eigenvectors
        coeffs = -a/(lam+sigma[...,np.newaxis])
        coeffscol = np.expand_dims(coeffs,axis=-1)
        
        p = np.matmul(u, coeffscol)
        p = np.squeeze(p, axis=-1)
        
        #compute predicted reduction in loss function from eigenvalues and eigenvectors
        predicted_reduction = -np.sum(a*coeffs + 0.5*lam*np.square(coeffs), axis=-1)
        
        at_boundary = np.logical_not(usesolu)
        
        return p, at_boundary, predicted_reduction, edm
        

        #return x_out, trust_radius_out, actual_reduction, predicted_reduction, rho, val, gradmag


@jax.jit
def sr1Update(B,y,dx):
    y = np.expand_dims(y, axis=-1)
    dx = np.expand_dims(dx, axis=-1)
    Bx = np.matmul(B,dx)
    dyBx = y - Bx
    dyBxT = np.swapaxes(dyBx,-1,-2)
    den = np.matmul(dyBxT,dx)
    denv = np.squeeze(den,axis=-1)
    dennorm = np.sqrt(np.sum(dx**2, axis=-2))*np.sqrt(np.sum(dyBx**2,axis=-2))
    dentest = np.less(np.abs(denv),1e-8*dennorm)
    dentest = dentest[...,np.newaxis]
    num = np.matmul(dyBx,dyBxT)
    print("num den dentest denv dennom dx dyBx", num.shape, den.shape, dentest.shape, denv.shape, dennorm.shape, dx.shape, dyBx.shape)
    num = np.where(dentest, 0., num)
    den = np.where(dentest, 1., den)
    den = np.where(np.equal(num,0.),1.,den)
    deltaB = num/den
    
    B = B + deltaB
    return B