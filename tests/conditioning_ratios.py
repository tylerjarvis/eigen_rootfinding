"""
Computes the conditioning ratios of random quadratics in dimensions
2-10
"""
from devastating_example_test_scripts import *
import eigen_rootfinding as er
from eigen_rootfinding.utils import condeigs, newton_polish
from eigen_rootfinding.polyroots import solve
from eigen_rootfinding.Multiplication import *
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import sys
from matplotlib import ticker as mticker
from scipy.stats import linregress
from matplotlib.patches import Patch
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter

macheps = 2.220446049250313e-16

def devastating_conditioning_ratios(dims,eps,kind,newton,N=50,just_dev_root=True,
                                seed=468,delta=0,save=True,verbose=0,detailed=False):
    """Computes the conditioning ratios of a system of polynomails.

    Parameters
    ----------
    dims : list of ints
        The dimensions to test
    eps : float
        epsilon value for the devastating example
    kind : string
        the type of devastating example system. One of 'power', 'spower',
        'cheb', and 'chebs'.
    newton : bool
        whether or not to newton polish the roots
    N : int or list
        number of tests to run in each dimension
    just_dev_root : bool
        If true, only returns conditioning ratios for the devastating root.
        Otherwise, returns conditioning ratios for all roots.
    seed : int
        random seed to use in generating the systems
    delta : float
        the amount by which to perturb the system
    save : bool
        whether to save and return the results or just return them
    verbose : int (default 0)
        the level of verbosity
    returns
    -------
    conditioning ratios: (dim, N, num_roots) or (dim, N) array
        Array of conditioning ratios. The [i,j] spot is  the conditioning ratio for
        the i'th coordinate in the j'th test system.
    """
    if verbose>0:print('Devastating Example in dimensions',dims)
    np.random.seed(seed)
    if isinstance(N,int):
        N = [N]*len(dims)
    crs = dict() #conditioning ratios dictionary
    if detailed:
        rcs = dict()
        ecs = dict()
    if kind in ['power','cheb']: shifted = False
    else: shifted = True
    for n,dim in zip(N,dims):
        if save:
            if newton: folder = 'conditioning_ratios/dev/newton/dim{}/'.format(dim)
            else:      folder = 'conditioning_ratios/dev/nopol/dim{}/'.format(dim)
        if verbose>0:print('Dimension', dim)
        cr = []
        if detailed:
            ec = []
            rc = []
        for _ in range(n):
            #get a random devastating example
            polys = randpoly(dim,eps,kind)
            if verbose>2: print('System Coeffs',*[p.coeff for p in polys],sep='\n')
            if delta > 0:
                polys = perturb(polys,delta)
            conditioning_ratio = conditioningratio(polys,dim,newton,dev=just_dev_root,shifted=shifted,verbose=verbose>1,detailed=detailed)
            if newton:
                if detailed:
                    conditioning_ratio, max_diff, smallest_dist_between_roots, eig_conds, root_conds = conditioning_ratio
                    if 10*max_diff >= smallest_dist_between_roots:
                        print('**Potentially converging roots with polishing**')
                        print('\tNewton changed roots by at most: {}'.format(max_diff))
                        print('\tDist between root was at least:  {}'.format(smallest_dist_between_roots))
                else:
                    conditioning_ratio, max_diff, smallest_dist_between_roots = conditioning_ratio
                    if 10*max_diff >= smallest_dist_between_roots:
                        print('**Potentially converging roots with polishing**')
                        print('\tNewton changed roots by at most: {}'.format(max_diff))
                        print('\tDist between root was at least:  {}'.format(smallest_dist_between_roots))
            elif detailed:
                    conditioning_ratio, eig_conds, root_cond = conditioning_ratio
                    ec.append(eig_conds)
                    rc.append(root_cond)
            if verbose>0:print(_+1,'done')
            cr.append(conditioning_ratio)
            if save:
                np.save(folder+'deg2_sys{}.npy'.format(_),cr)
                if verbose>0:print(_+1,'saved')
        crs[dim] = np.array(cr)
        if detailed:
            rcs[dim] = np.array(rc)
            ecs[dim] = np.array(ec)
        if save: np.save(folder+'deg2.npy',crs[dim])
    if detailed: return crs, ecs, rcs
    else: return crs

def find_root_idx(roots,root):
    dists = [mp.norm(roots[root_num,:].T - root) for root_num in range(roots.rows)]
    return np.argmin(dists)

def conditioningratio(polys,dim,newton,dev=False,shifted=None,root=None,verbose=False,detailed=False):
    """Computes the conditioning ratios of a system of polynomails.

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomial system
    dim : int
        dimension of the polynomials
    newton : bool
        whether or not to newton polish the roots
    dev : bool
        whether or not we are computing the conditioning ratio for a devastating
        example system, in which case we want to use the root at the origin
    shifted : bool
        for devastating systems, whether the system is
    root : 1d nparray
        optional parameter for when you know the actual
        root you want to find the conditioning ratio of
    returns
    -------
    conditioning ratios: (dim, num_roots) array
        Array of conditioning ratios. The [i,j] spot is  the conditioning ratio for
        the i'th coordinate of the j'th root.
    """
    roots,M = solve(polys,max_cond_num=np.inf,verbose=verbose,return_mult_matrices=True)
    if newton:
        dist_between_roots = la.norm(roots[:,np.newaxis]-roots,axis=2)
        smallest_dist_between_roots = np.min(dist_between_roots[np.nonzero(dist_between_roots)])
        newroots = np.array([newton_polish(polys,root,tol=10*macheps) for root in roots])
        max_diff = np.max(np.abs(newroots-roots))
        roots = newroots
    #find the conditioning ratios for all the roots
    if root is not None:
        #find the root
        idx = find_root_idx(roots,root)
        #compute eigenvalue condition numbers
        eig_conds = []
        for d in range(dim):
            M_ = M[d]
            vals, vecR = mp.eig(M_)
            eig_conds_curr = condeigs(M_,vals,vecR)
            arr = sort_eigs(vals,roots[:,d],arr=True)
            eig_conds.append([eig_conds_curr[int(k)] for k in arr][idx])
        #compute the condition numbers of the roots
        J = mp.matrix(dim)
        for j,poly in enumerate(polys):
            grad = poly.grad(root)
            for k in range(dim):
                J[j,k] = grad[k]
        S = mp.svd(J,compute_uv=False)
        root_cond = 1/S[S.rows-1]
        #compute the conditioning ratios
        ratios = [eig_cond/root_cond for eig_cond in eig_conds]
        #truncate results into numpy arrays
        ratios = np.array(ratios,dtype=np.float64)
        if detailed:
            eig_conds = np.array(eig_conds,dtype=np.float64)
            root_cond = np.array(root_cond,dtype=np.float64)
            if newton: return ratios, max_diff, smallest_dist_between_roots, eig_conds, root_cond
            else: return ratios, eig_conds, root_cond
        else:
            if newton: return ratios, max_diff, smallest_dist_between_roots
            else: return ratios
    elif not dev:
        eig_conds = []#np.empty((dim,len(roots)))
        for d in range(dim):
            M_ = M[d]
            vals, vecR = mp.eig(M_)
            eig_conds[d] = condeigs(M_,vals,vecR)
            arr = sort_eigs(vals,roots[:,d],arr=True)
            vals = vals[arr]
            eig_conds.append([eig_conds_curr[int(k)] for k in arr])
        #compute the condition numbers of the roots
        root_conds = []
        for i,root in enumerate(roots):
            for j,poly in enumerate(polys):
                grad = poly.grad(root)
                for k in range(dim):
                    J[j,k] = grad[k]
            S = mp.svd(J,compute_uv=False)
            root_cond = 1/S[S.rows-1]
            root_conds[i] = root_cond
        #compute the conditioning ratios
        ratios = [[eig_cond/root_cond for eig_cond in eig_conds_curr]
                    for eig_conds_curr, root_cond in zip(eig_conds,root_conds)]
        #truncate the results into numpy array
        ratios = np.array(ratios,dtype=np.float64)
        if detailed:
            eig_conds = np.array(eig_conds,dtype=np.float64)
            root_conds = np.array(root_conds,dtype=np.float64)
            if newton: return ratios, max_diff, smallest_dist_between_roots, eig_conds, root_conds
            else: return ratios, eig_conds, root_conds
        else:
            if newton: return ratios, max_diff, smallest_dist_between_roots
            else: return ratios
    #only find the conditioning ratio for the root at the origin
    else:
        #find the root at the origin
        if shifted:
            dev_root = mp.ones(dim,1)
        else:
            dev_root = mp.zeros(dim,1)
        idx = find_root_idx(roots,dev_root)
        #compute eigenvalue condition numbers
        eig_conds = []
        for d in range(dim):
            M_ = M[d]
            vals, vecR = mp.eig(M_)
            eig_conds_curr = condeigs(M_,vals,vecR)
            arr = sort_eigs(vals,roots[:,d],arr=True)
            eig_conds.append([eig_conds_curr[int(k)] for k in arr][idx])
        #compute the condition numbers of the root
        J = mp.matrix(dim)
        for j,poly in enumerate(polys):
            grad = poly.grad(dev_root)
            for k in range(dim):
                J[j,k] = grad[k]
        S = mp.svd(J,compute_uv=False)
        root_cond = 1/S[S.rows-1]
        #compute the conditioning ratios
        ratios = [eig_cond/root_cond for eig_cond in eig_conds]
        #truncate results into numpy arrays
        ratios = np.array(ratios,dtype=np.float64)
        if detailed:
            eig_conds = np.array(eig_conds,dtype=np.float64)
            root_cond = np.array(root_cond,dtype=np.float64)
            if newton: return ratios, max_diff, smallest_dist_between_roots, eig_conds, root_cond
            else: return ratios, eig_conds, root_cond
        else:
            if newton: return ratios, max_diff, smallest_dist_between_roots
            else: return ratios

def get_conditioning_ratios(coeffs, newton, save=True):
    """Computes the conditioning ratios of a bunch of systems of polynomails.

    Parameters
    ----------
    coeffs : (N,dim,deg,deg,...) array
        Coefficient tensors of N test systems. Each test system should have dim
        polynomial systems of degree deg
    newton : bool
        whether or not to newton polish the roots
    save : bool
        whether or not to save and return the results or just return them
    returns
    -------
    conditioning ratios: (N, dim, deg^dim) array
        Array of conditioning ratios. The [k,i,j] spot is  the conditioning ratio for
        the i'th coordinate of the j'th root of the k'th system
    """
    N,dim = coeffs.shape[:2]
    deg = 2
    print((N,dim,deg**dim))
    not_full_roots = np.zeros(N,dtype=bool)
    crs = [0]*N
    if save:
        if newton: folder = 'conditioning_ratios/rand/newton/dim{}/'.format(dim)
        else:      folder = 'conditioning_ratios/rand/nopol/dim{}/'.format(dim)
    for i,system in enumerate(coeffs):
        polys = [er.MultiPower(c) for c in system]
        cr = conditioningratio(polys,dim,newton)
        if newton:

            cr,max_diff,smallest_dist_between_roots = cr
            if not 10*max_diff < smallest_dist_between_roots:
                print('**Potentially converging roots with polishing**')
                print('\tNewton changed roots by at most: {}'.format(max_diff))
                print('\tDist between root was at least:  {}'.format(smallest_dist_between_roots))
        #only records if has the right number of roots_sort
        print(i+1,'done')
        if cr.shape[1] == deg**dim:
            crs[i] = cr
            if save: np.save(folder+'deg2_sys{}.npy'.format(i),cr)
        else:
            not_full_roots[i] = True
            if save: np.save(folder+'not_full_roots_deg2.npy',not_full_roots)
        if save: print(i+1,'saved')
    #final save at the end
    if save:
        np.save(folder+'deg2_res.npy',crs)
        print('saved all results')
    return crs

'''functions to generate random systems that almost have double roots.

get_scalar, get_coeff and get_MultiPower can be used to find a hyperellipse/hyperbola
with pre-chosen roots.

the rest of the functions use specially chosen roots to generate examples.
'''
def get_scalar(center,roots):
    'solves for the scalars in the conic equation. see conditioning_ratios.ipynb for details'
    dim = roots.shape[1]
    RHS = np.ones(dim)
    return la.solve((roots - center)**2,RHS)

def get_coeff(center,roots):
    """
    finds the coefficient tensor of the hyperellipses/hyperbolas with specified center
    and roots
    """
    scalar = get_scalar(center,roots)
    dim = len(center)
    coeff = np.zeros([3]*dim)
    spot = [0]*dim
    coeff[tuple(spot)] = np.sum(scalar*center**2)-1
    for var,c,s in zip(range(dim),center,scalar):
        #linear
        spot[var] = 1
        coeff[tuple(spot)] = -2*s*c
        spot[var] = 2
        coeff[tuple(spot)] = s
        spot[var] = 0
    return coeff

def get_MultiPower(center,roots):
    """
    creates a MultiPower object of a hyperellipse/hyperbola with a specified center and roots
    """
    return er.MultiPower(get_coeff(center,roots))

def gen_almost_multiple_roots(dim,seed,alpha,verbose=False):
    """
    Generates an n-dimensional hyperellipse/hyperbola with random seed 'seed.'
    The first root is *almost* multiplicity 'dim.' Specifically, the first root is chosen,
    and then dim-1 perturbations of that root are forced to also be roots. Those perturbations
    are chosen using a normal distribution in each coordinate with mean 0 and standard deviation alpha.
    """
    np.random.seed(seed)
    centers = np.random.randn(dim,dim)
    if verbose: print('Centers:',centers,sep='\n')
    root = np.random.randn(dim)
    if verbose: print('Root:',root,sep='\n')
    dirs = np.random.randn(dim,dim)*alpha
    dirs[0] = 0
    if verbose: print('Directions:',dirs,sep='\n')
    roots = root+dirs
    if verbose: print('Roots:',roots,sep='\n')
    return roots,[get_MultiPower(c,roots) for c in centers]

def gen_almost_double_roots(dim,seed,alpha,verbose=False):
    """
    Generates an n-dimensional hyperellipse/hyperbola with random seed 'seed.'
    The first root is *almost* a double root. Specifically, the first root is chosen,
    and then a perturbation of that root is forced to also be a root. The perturbation
    is chosen using a normal distribution in each coordinate with mean 0 and standard deviation alpha.
    There are also dim-2 other randomly pre-chosen chosen real roots. To see what they are, usee verbose=True.
    """
    np.random.seed(seed)
    centers = np.random.randn(dim,dim)
    if verbose: print('Centers:',centers,sep='\n')
    root = np.random.randn(1,dim)
    if verbose: print('Root:',root,sep='\n')
    direction = np.random.randn(1,dim)*alpha
    if verbose: print('Perturbation:',direction,sep='\n')
    root2 = root+direction
    if verbose: print('Almost Double Root:',root2,sep='\n')
    if dim > 2:
        other_roots = np.random.randn(dim-2,dim)
        if verbose: print('Other Roots:',other_roots,sep='\n')
        roots = np.vstack((root,root2,other_roots))
    else:
        roots = np.vstack((root,root2))
    if verbose: print('Total Roots:',roots,sep='\n')
    return roots,[get_MultiPower(c,roots) for c in centers]

def gen_rand_hyperconic(dim,seed,alpha,verbose=False):
    """
    Generates an n-dimensional hyperellipse/hyperbola with random seed 'seed.'
    There are dim randomly pre-chosen real roots. To see what they are, usee verbose=True.
    """
    np.random.seed(seed)
    centers = np.random.randn(dim,dim)
    if verbose: print('Centers:',centers,sep='\n')
    roots = np.random.randn(dim,dim)
    if verbose: print('Roots:',roots,sep='\n')
    return roots,[get_MultiPower(c,roots) for c in centers]

def get_data(alpha,gen_func,seeds = {2:range(300),3:range(300),4:range(300)},detailed=False,save=False,filename=None,filenameextension=''):
    """
    Computes the conditioning ratio of the first generated root of systems generated with gen_func(dim,seed,alpha) for each
    seed in the seeds dictionary.
    Seeds is assumed to be a dictionary where the keys are the dimensions you want to test in, and the values
    are an iterable of random seeds to generate random systems with.
    """
    dims = seeds.keys()
    data = {d:[] for d in dims}
    root_conds = {d:[] for d in dims}
    eig_conds = {d:[] for d in dims}
    if save:
        if gen_func.__name__ == 'gen_almost_double_roots':
            test_type = 'doub'
        elif gen_func.__name__ == 'gen_almost_multiple_roots':
            test_type = 'mult'
        elif gen_func.__name__ == 'gen_rand_hyperconic':
            test_type = 'rand'
        foldername = 'conditioning_ratios/nearby_roots/'+test_type+'/'
        if filename is None:
            filename = 'alpha{}'.format(str(alpha).replace(".", "_"))
    for dim in dims:
        print('dim',dim)
        for n in seeds[dim]:
            print('seed',n)
            roots,polys = gen_func(dim=dim,seed=n,alpha=alpha)
            cr = conditioningratio(polys,dim,newton=False,root=mp.matrix(roots[0]),detailed=detailed)
            if detailed:
                cr, eig_cond, root_cond = cr
                root_conds[dim].append(root_cond)
                eig_conds[dim].append(eig_cond)
            data[dim].extend(cr)
            if save:
                if dim >= 4:
                    print('saving...')
                    dim_seed_marker = '_{}D_seed{}'.format(dim,n)
                    np.save(foldername+filename+dim_seed_marker+filenameextension, np.float64(cr))
                    if detailed:
                        np.save(foldername+filename+dim_seed_marker+filenameextension+'_eigcond', np.float64(eig_cond))
                        np.save(foldername+filename+dim_seed_marker+filenameextension+'_rootcond', np.float64(root_cond))
        data[dim] = np.array(data[dim]).flatten()
        if detailed:
            root_conds[dim] = np.array(root_conds[dim]).flatten()
            eig_conds[dim] = np.array(eig_conds[dim]).flatten()
        if save:
            if dim < 4:
                print('saving...')
                dim_marker = '_{}D'.format(dim)
                np.save(foldername+filename+dim_marker+filenameextension, data)
                if detailed:
                    np.save(foldername+filename+dim_marker+filenameextension+'_eigconds', eig_conds)
                    np.save(foldername+filename+dim_marker+filenameextension+'_rootconds', root_conds)
    if save:
        print('saving final results...')
        np.save(foldername+filename+filenameextension, data)
        if detailed:
            np.save(foldername+filename+filenameextension+'_eigconds', eig_conds)
            np.save(foldername+filename+filenameextension+'_rootconds', root_conds)
        print('done with alpha = {}!'.format(alpha))
    if detailed: return data,eig_conds,root_conds
    else: return data

def plot(datasets,labels=None,yaxislabel='Conditioning Ratio',subplots=None,title=None,filename='conditioning_ratio_plot',figsize=(6,4),
         dpi=400,best_fit=True,_2nd_plot=None, min_ylim=None, max_ylim=None):
    """
    Plots conditioning ratio data.

    Parameters
    ----------
    datasets : list of dictionaries
        Conditioning ratio datasets to plot. Each dataset dictionary should be
        formatted to map dimension to an array of conditioning ratios
    digits_lost : bool
        whether the y-axis should be a log scale of the conditioning ratios
        or a linear scale of the digits lost
    figsize : tuple of floats
        figure size
    dpi : int
        dpi of the image
    """
    if subplots is None: fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize,dpi=dpi)
    #else: fig, ax = plt.subplots(nrows=subplots[0], ncols=subplots[1], figsize=figsize,dpi=dpi,sharey=True,sharex=True)
    else: fig, ax = plt.subplots(nrows=subplots[0], ncols=subplots[1], figsize=figsize,dpi=dpi,sharey=False,sharex=False)
    def plot_dataset(ax,data,color,label=None):
        pos = 2+np.arange(len(data))
        #log before plot
        data_log10 = [np.log10(data[d].flatten()) for d in data.keys()]
        #violins
        parts = ax.violinplot(data_log10,
                      positions=pos,
                      widths=.8,
                      points=1000,
                      showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(.3)
        #boxplots
        maxs = [np.max(g) for g in data_log10]
        mins = [np.min(g) for g in data_log10]
        ax.hlines(maxs,pos-.02,pos+.02,lw=1)
        ax.hlines(mins,pos-.02,pos+.02,lw=1)
        ax.vlines(pos,mins,maxs,lw=.5,linestyles='dashed')
        box_props = dict(facecolor='w',color=color)
        median_props = dict(color=color)
        box = ax.boxplot(data_log10,positions=pos,
                   vert=True,
                   showfliers=False,
                   patch_artist=True,
                   boxprops=box_props,
                   widths=.35,
                   medianprops=median_props)
        plt.setp(box['whiskers'], color=color)
        plt.setp(box['caps'], color=color)
        if best_fit:
            points = np.array([[d,val] for i,d in enumerate(data.keys()) for val in data_log10[i]])
            slope, intercept = linregress(points)[:2]
            growth_rate = 10**slope-1
            if label is not None:
                print(label)
            print('Slope:',slope, '\nIntercept:',intercept,'\nExponential Growth Rate:',growth_rate,end='\n\n')
            ax.plot(pos,pos*slope+intercept,c=color)
    if subplots is None:
        dims = 2+np.arange(np.max([len(data.keys()) for data in datasets]))
        ax.yaxis.grid(color='gray',alpha=.15,linewidth=1,which='major')
        if labels is None:
            for i,dataset in enumerate(datasets):
                plot_dataset(ax,dataset,f'C{i}')
        else:
            for i,dataset in enumerate(datasets):
                plot_dataset(ax,dataset,f'C{i}',labels[i])
        ax.set_title('Conditioning Ratios of Quadratic Polynomial Systems')
        ax.set_ylabel(yaxislabel)
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        if min_ylim is not None:
            ax.yaxis.set_ticks([np.log10(x) for p in range(min_ylim,max_ylim)
                               for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)
        ax.set_xlabel('Dimension')
        legend_elements = [Patch(facecolor=f'C{i}') for i in range(len(datasets))]
        ax.legend(legend_elements,labels)
        if title is None:
            ax.set_title('Conditioning Ratios of Quadratic Polynomial Systems')
        else:
            ax.set_title(title)
    else:  # for subplots ##################################################################
        for ax_,datasets_axis,title_axis,labels_axis in zip(ax,datasets,title,labels):
            ax_.yaxis.grid(color='gray',alpha=.15,linewidth=1,which='major')
            if labels is None:
                for i,dataset in enumerate(datasets_axis):
                    plot_dataset(ax_,dataset,f'C{i}')
            else:
                for i,dataset in enumerate(datasets_axis):
                    plot_dataset(ax_,dataset,f'C{i}',labels_axis[i])
            ax_.set_title('Conditioning Ratios of Quadratic Polynomial Systems')
            ax_.set_xlabel('Dimension')
            legend_elements = [Patch(facecolor=f'C{i}') for i in range(len(datasets_axis))]
            ax_.legend(legend_elements,labels_axis)
            if title is None:
                ax_.set_title('Conditioning Ratios of Quadratic Polynomial Systems')
            else:
                ax_.set_title(title_axis)
        if title is not None: plt.suptitle(title[-1])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        dims = 2+np.arange(np.max([len(data.keys()) for data in datasets[0]]))
        ax[0].set_ylabel('Conditioning Ratio')
        ax[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        if min_ylim is not None:
            ax[0].yaxis.set_ticks([np.log10(x) for p in range(min_ylim,max_ylim)
                               for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)
        # insert slopes subplot stuff here ####################################################
        #######################################################################################
        if _2nd_plot is not None:
            ax[1].clear()
            ax[1].semilogy(_2nd_plot[0], _2nd_plot[1])
            ax[1].set_xlabel(r'Standard Deviation of Perturbation, $\alpha$')
            ax[1].set_ylabel('Growth Rate, $r$')
            ax[1].set_title(title[1])
            ax[1].set_xscale('log')
            ax[1].xaxis.set_ticks([x for p in range(-6,-1)
                               for x in np.linspace(10**p, 10**(p+1), 10)],minor=True)
            ax[1].xaxis.set_ticks([10**p for p in range(-6,0)],minor=False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fname='figures/'+filename+'.pdf',bbox_inches='tight',dpi=dpi,format='pdf')
    plt.show()

if __name__ == "__main__":
    #for running random and devastating systems on the server
    #INPUT FORMAT test_type--dev or rand; newton_polish--newton or nopol; dims-- dimensions to run in
    # input = sys.argv[1:]
    # test = input[0]
    # if input[1] == 'newton':
    #     newton = True
    # elif input[1] == 'nopol':
    #     newton=False
    # else:
    #     raise ValueError("2nd input must be one of 'newton' for polishing or 'nopol' for no polishing")
    # dims = [int(i) for i in input[2:]]
    # if test == 'rand':
    #     for dim in dims:
    #         coeffs = np.load('random_coeffs/dim{}_deg2_randn.npy'.format(dim))
    #         crs = get_conditioning_ratios(coeffs, newton)
    # elif test == 'dev':
    #     eps = .1
    #     kind = 'power'
    #     N = 50
    #     devastating_conditioning_ratios(dims,eps,kind,newton,N=N)
    # else:
    #     raise ValueError("1st input must be one of 'rand' for random polys 'dev' for devastating example")

    #for running nearly multiple roots on the server
    input = sys.argv[1:]
    test = input[0]
    digits_precision = input[1]
    print(digits_precision)
    mp.mp.dps = int(digits_precision)
    print(mp.mp)
    seed_min = int(input[2])
    seed_max = int(input[3])
    num_tests_per_dim = seed_max - seed_min
    alpha_vals = input[4:]
    for alpha in alpha_vals:
        alpha = np.float(alpha)
        if test == 'mult':
            gen_func = gen_almost_multiple_roots
        elif test == 'doub':
            gen_func = gen_almost_double_roots
        else:
            raise ValueError("'test' must be one of 'mult', 'doub'")
        get_data(alpha,
                 gen_func,
                 seeds = {2:range(seed_min,seed_max),3:range(seed_min,seed_max),4:range(seed_min,seed_max)},
                 detailed=True,
                 save=True,
                 filename=None,
                 filenameextension='_{}testsperdim_{}digitsprecision'.format(num_tests_per_dim,digits_precision))
