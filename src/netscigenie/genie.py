import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from bs4 import BeautifulSoup
import requests


def degree_distribution(G, number_of_bins=15, log_binning=True, density=True, directed=False):
    """
    Given a degree sequence, return the y values (probability) and the
    x values (support) of a degree distribution that you're going to plot.
    
    Parameters
    ----------
    G (nx.Graph):
        the network whose degree distribution to calculate

    number_of_bins (int):
        length of output vectors
    
    log_binning (bool):
        if you are plotting on a log-log axis, then this is useful
    
    density (bool):
        whether to return counts or probability density (default: True)
        Note: probability densities integrate to 1 but do not sum to 1. 

    directed (bool or str):
        if False, this assumes the network is undirected. Otherwise, the
        function requires an 'in' or 'out' as input, which will create the 
        in- or out-degree distributions, respectively.
        
    Returns
    -------
    bins_out, probs (np.ndarray):
        probability density if density=True node counts if density=False; binned edges
    
    """

    if directed:
        if directed=='in':
            k = list(dict(G.in_degree()).values()) # get the in degree of each node
        elif directed=='out':
            k = list(dict(G.out_degree()).values()) # get the out degree of each node
        else:
            out_error = "Help! if directed!=False, the input needs to be either 'in' or 'out'"
            print(out_error)
            return out_error
    else:
        k = list(dict(G.degree()).values()) # get the degree of each node


    kmax = np.max(k)    # get the maximum degree
    kmin = 0            # let's assume kmin must be 0

    if log_binning:
        bins = np.logspace(0, np.log10(kmax+1), number_of_bins+1)
    else:
        bins = np.linspace(0, kmax+1, num=number_of_bins+1)
    probs, _ = np.histogram(k, bins, density=density)
    bins_out = bins[1:] - np.diff(bins)/2.0
    
    return bins_out, probs


def average_clustering_by_degree(G):
    """
    Calculate the average clustering coefficient of nodes of degree k (C(k))
    for a NetworkX graph G.

    Inputs:
    G (networkx.Graph): A NetworkX graph

    Outputs:
    clustering_k (dict): A dictionary where the keys are degrees k and the values are the average clustering coefficient of nodes of degree k.
    """
    
    clustering_sum = defaultdict(float)
    degree_count = defaultdict(int)
    
    for node in G.nodes():
        k = G.degree(node)
        clustering = nx.clustering(G, node)
        
        # Update the sum of clustering coefficients for degree k nodes
        clustering_sum[k] += clustering
        degree_count[k] += 1
    
    clustering_k = {k: clustering_sum[k] / degree_count[k] for k in degree_count}
    
    return clustering_k


def average_neighbor_degree_by_degree(G):
    """
    Calculate the average degree of neighbors of nodes of degree k for a NetworkX graph G.

    Inputs:
    G (networkx.Graph): A NetworkX graph

    Outputs:
    knn_k (dict): A dictionary where the keys are degrees k and the values are the average degree of the neighbors of nodes of degree k.
    """
    
    degree_neighbor_sum = defaultdict(float)
    degree_count = defaultdict(int)
    
    for node in G.nodes():
        k = G.degree(node)
        
        neighbor_degrees = [G.degree(neighbor) for neighbor in G.neighbors(node)]
        
        # Calculate the average degree of the neighbors (if the node has neighbors)
        if len(neighbor_degrees) > 0:
            avg_neighbor_degree = sum(neighbor_degrees) / len(neighbor_degrees)
        else:
            avg_neighbor_degree = 0
        
        # Update the sum of neighbors' degrees for degree k nodes
        degree_neighbor_sum[k] += avg_neighbor_degree
        degree_count[k] += 1
    
    knn_k = {k: degree_neighbor_sum[k] / degree_count[k] for k in degree_count}
    
    return knn_k


def log_binning_function(x,y,B, xtype):
    """
    A function that does log-binning on array x and y.
    Inputs: x--xvalues in a list
           y--values in a list 
           B--Number of bins
           xtype-- "left" to consider the left boundary of the bin or "centre"  
    Outputs: X and Y, the logbinned values of x and y.

    Thanks to Narayan who helped with this function in NetSci II class!
    """
    Lf = (np.log(np.max(x)) - np.log(np.min(x)))/B
    bi = [np.exp(np.log(np.min(x)) + (i-1)*Lf) for i in range(1,B+2)]

    if xtype == "left":
        X = bi[:-1]
    if xtype == "centre":
        X = [np.sqrt(bi[i]*bi[i+1]) for i in range(len(bi)-1)]
    
    xp = bi
    xp[-1] = xp[-1]+1

    df = pd.DataFrame()
    df["x"] = x 
    df["y"] = y 
    
    Y = [df[(df['x'] < xp[j+1]) & (df['x'] >= xp[j])]["y"].mean() for j in range(len(xp)-1)]

    return X,Y

def log_binning_for_distributions(x,B):
    """ A function that returns the log-binning of an array x

    Inputs:
    x (array): the input array to be log-binned
    B (int): number of bins

    Outputs:
    X (array): positions of centers of bins
    y (array): log binned y-values.
    """
    n = len(x)
    q1 = np.log(min(x))
    D = np.log(max(x))-np.log(min(x))
    L = D/B # length of each bin in log space
    X, y, s = np.zeros(B), np.zeros(B), np.zeros(B) # these are the X_i's, Y_i's and number of elements in each bin respectively
    q = np.zeros(B+1) # these are the q_i's (equally spaced in log space)
    for i in range(0,B+1):
        q[i] = q1 + i*L
    b = np.exp(q)

    for i in range(0,B):
        X[i] = np.exp((q[i]+q[i+1])/2)

    y = np.histogram(x,bins=b, density=True)[0]

    plt.scatter(X,y)
    plt.xscale('log')
    plt.yscale('log')
    return X, y

def pareto_sample_inverse_CDF(size, k_bar, gamma):
    """ A function that returns a random sample from Pareto distribution using the Inverse CDF method.

    Inputs:
    size (int): size of the sample
    k_bar, gamma (float): parameters in Pareto distribution.

    Outputs:
    samples (np.array): The random sample of size n.
    """
    u = np.random.uniform(0, 1, size)
    kappa_0 = k_bar * (gamma-2)/(gamma-1)
    samples = kappa_0 / np.power(1-u, 1/(gamma-1))
    return samples

def HSCM_network(n, k_bar, gamma):
    """ A function that returns a Hypersoft Configuration Model graph.

    Inputs:
    n (int): size of the network
    k_bar (float): average degree of network
    gamma (float): degree exponent in degree distribution

    Outputs:
    G (nx.Graph): The desired graph.
    """
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    kappas_array = pareto_sample_inverse_CDF(n, k_bar, gamma)
    # print(kappas_array)
    # connection probability
    for i in range(n):
        for j in range(i+1, n):
            expression = (k_bar * n)/(kappas_array[i] * kappas_array[j])
            p = 1/(1+expression)
            if np.random.uniform(0,1) < p:
                G.add_edge(i, j)
    # nx.draw(G, with_labels=True, node_size=100)
    # plt.show()
    return G, kappas_array

def HER_network(n, k_bar, gamma):
    """ A function that returns a Hypercanonical Erdos-Renyi graph.

    Inputs:
    n (int): size of the network
    k_bar (float): average degree of network
    gamma (float): degree exponent in degree distribution

    Outputs:
    G (nx.Graph): The desired graph.
    """
    
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    kappa = pareto_sample_inverse_CDF(1, k_bar, gamma)
    p = min(1, kappa/n)
    # print(p)
    # connection probability
    for i in range(n):
        for j in range(i+1, n):
            if np.random.uniform(0,1) < p:
                G.add_edge(i, j)
    # nx.draw(G, with_labels=True, node_size=100)
    # plt.show()
    return G

### SCRAPING NEU DEPARTMENTS WEBSITE ###


def get_NEU_department_hrefs():
    catalog_res = requests.get('https://catalog.northeastern.edu/course-descriptions/')
    catalog_html = catalog_res.text
    soup = BeautifulSoup(catalog_html)

    soup_alpha_div = soup.find('div', id="atozindex")

    department_hrefs = []
    for ul in soup_alpha_div.find_all('ul'):
        for li in ul.find_all('li'):
            department_hrefs.append(li.a.get('href'))

    return department_hrefs

def get_prerequisites(dept_html):
    """
    A function that returns names of prerequisites for each course in a department's course catalog.
    
    Inputs: 
    dept_html (str): a string of raw html

    Outputs:
    catalog (dict): where the keys are course numbers and the values are lists of prerequisites.
    """
    soup = BeautifulSoup(dept_html)
    course_sections = soup.find_all('div', class_='courseblock')

    catalog = {}

    for course in course_sections:
        course_name = course.find('p', class_='courseblocktitle noindent').get_text()
        course_name = course_name.replace('\xa0',' ') # replace non-breaking space
        course_name = course_name.replace('  ', ' ') # remove extra spaces
        course_name = course_name[0:course_name.find('.')] # keep only the course number 
        catalog[course_name] = []

        prereqs_sections = course.find_all('p', class_="courseblockextra noindent")
        prereqs = []
        for prereq_section in prereqs_sections:
            pre_text = prereq_section.find("strong")
            pre_text = pre_text.text
            pre_text = pre_text.strip()

            if pre_text == 'Prerequisite(s):':
                temp = prereq_section.find_all('a', class_="bubblelink code")
                lst = [temp[i]['title'].replace('\xa0',' ') for i in range(len(temp))]
                prereqs = list(set(lst))
                catalog[course_name] = catalog[course_name] + prereqs

    return catalog

def get_prerequisites_per_course(dept_html):
    """
    A function that returns the ratio of prerequisites to the number of courses for each department.
    
    Inputs: 
    dept_html (str): a string of raw html for the department 

    Outputs:
    ratio (float): 
    """
    soup = BeautifulSoup(dept_html)
    course_sections = soup.find_all('div', class_='courseblock')

    num_courses = len(course_sections)
    num_prereqs = 0

    for course in course_sections:
        prereqs_sections = course.find_all('a', class_="bubblelink code")
        prereqs = []
        for prereq_section in prereqs_sections:
            temp = prereq_section['title']
            temp = temp.replace('\xa0',' ') # replace non-breaking space
            prereqs.append(temp)

        prereqs = list(set(prereqs))
        num_prereqs += len(prereqs)

    if num_courses != 0:
        ratio = num_prereqs / num_courses
    else:
        ratio = 0
        
    return ratio

def add_num(x, y):
    return x + y

