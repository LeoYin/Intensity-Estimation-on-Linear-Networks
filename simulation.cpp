q#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "armadillo"

using namespace std;
using namespace arma;

typedef pair<int, int> edge_pair;

//Poisson Point Process
struct PPP_node
{
	arma::vec s, x, beta;
	double intensity;
	int index;

	void get_s();
	void get_x();
	void get_beta();
	void get_intensity();

	void initialize()
	{
		get_s();
		get_x();
		get_beta();
		get_intensity();
	}
};

//initialize s, x, beta of PPP
void PPP_node::get_s()
{
	s = arma::randu<vec>(2);
}

void PPP_node::get_x()
{
	x = arma::zeros<vec>(2);
	x(0) = 3.25 - abs(0.5 - s(0)) * 6;
	x(1) = 3.25 - abs(0.5 - s(1)) * 6;
}

void PPP_node::get_beta()
{
	beta = arma::zeros<vec>(2);
	beta(0) = ceil((abs(0.5 - s(0)) + 0.001) * 6);
	beta(1) = ceil((abs(0.5 - s(1)) + 0.001) * 6);
}

void PPP_node::get_intensity()
{
	intensity = exp((x.t() * beta).min());
}

//struct of graph
struct Graph
{
	int n_vertex, n_edge;
	vector<pair<double, edge_pair>> edges;

	Graph(int n_vertex, int n_edge)
	{
		this->n_vertex = n_vertex;
		this->n_edge = n_edge;
	}

	void add_edge(int u, int v, double w)
	{
		edges.push_back({ w,{ u, v } });
	}

	Graph MST();
	arma::mat get_mat();
};

//MST
struct disjoint_sets
{
	int *parent, *rnk;
	int n;

	disjoint_sets(int n)
	{
		this->n = n;
		parent = new int[n + 1];
		rnk = new int[n + 1];

		for (int i = 0; i <= n; i++)
		{
			rnk[i] = 0;
			parent[i] = i;
		}
	}

	int find(int u)
	{
		if (u != parent[u])
			parent[u] = find(parent[u]);
		return parent[u];
	}

	void merge(int x, int y)
	{
		x = find(x);
		y = find(y);

		if (rnk[x] > rnk[y])
			parent[y] = x;
		else
			parent[x] = y;
		if (rnk[x] == rnk[y])
			rnk[y]++;
	}
};

Graph Graph::MST()
{
	Graph g_MST(n_vertex, n_vertex - 1);

	sort(edges.begin(), edges.end());

	disjoint_sets ds(n_vertex);

	vector<pair<double, edge_pair>>::iterator it;
	for (it = edges.begin(); it != edges.end(); it++)
	{
		int u = it->second.first;
		int v = it->second.second;

		int set_u = ds.find(u);
		int set_v = ds.find(v);

		if (set_u != set_v)
		{
			g_MST.add_edge(u, v, it->first);
			ds.merge(set_u, set_v);
		}
	}

	return g_MST;
}

//map a graph into a matrix
arma::mat Graph::get_mat()
{
	arma::mat H = arma::zeros<mat>(n_edge, n_vertex);
	vector<pair<double, edge_pair>>::iterator it;
	int i = 0;
	for (it = edges.begin(); it != edges.end(); it++)
	{
		H(i, it->second.first) = 1;
		H(i, it->second.second) = -1;
		i++;
	}
	return H;
}

Graph map_generate(vector<PPP_node> u)
{
	vector<PPP_node>::iterator it1, it2;
	Graph g(u.size(), 0);
	vector<pair<double, int>> edge_find;
	vector<pair<double, int>>::iterator it3;
	double r;

	for (it1 = u.end() - 1; u.size() > 1; it1--, u.pop_back())
	{
		for (it2 = u.begin(); it2 != it1; it2++)
		{
			r = sqrt(pow(it1->s(0) - it2->s(0), 2) + pow(it1->s(1) - it2->s(1), 2));
			edge_find.push_back({ r, it2->index });
		}
		sort(edge_find.begin(), edge_find.end());
		if (u.size() > 3)
			for (it3 = edge_find.begin(); it3 != edge_find.begin() + 3; it3++)
				g.add_edge(it1->index, it3->second, it3->first);
		else
			for (it3 = edge_find.begin(); it3 != edge_find.end(); it3++)
				g.add_edge(it1->index, it3->second, it3->first);
		edge_find.clear();
	}
	g.n_edge = g.edges.size();

	return g;
}

//H to H_tilde
arma::mat diagblock(arma::mat A, int p)
{
	int nr = A.n_rows, nc = A.n_cols;
	arma::mat B = arma::zeros(p * nr, p * nc);
	for (int i=0; i < p; i++)
		B.submat(i * nr, i * nc, (i + 1) * nr -1, (i + 1) * nc -1) = A;
	return B;
}

//divide the domain of s into small pixels
arma::mat pixel_generate(int m, int n, vector<PPP_node> u)
{
	int i, j;
	vector<PPP_node>::iterator it;
	arma::mat A = arma::zeros<mat>(m, n);

	for (it = u.begin(); it != u.end(); it++)
	{
		i = floor(it->s(1) * 10);
		j = floor(it->s(0) * 10);
		A(i, j)++;
	}
	return A;
}

//soft thresholding operator
arma::vec soft_thresholding(double kappa, arma::vec a)
{
	int n = a.n_rows;
	arma::vec st = arma::zeros(n);
	for (int i = 0; i < n; i++)
		st(i) = (((1 - kappa / abs(a(i))) > 0) ? (1 - kappa / abs(a(i))) : 0)*a(i);
	return st;
}

//ADMM
struct ADMM_iterator
{
	int n, p;
	arma::vec b, z, u, Hb, dl;
	arma::mat ddl;

	ADMM_iterator(int n, int p)
	{
		this->n = n;
		this->p = p;
	}

	void initial()
	{
		b = arma::ones<vec>(n * p);
		z = arma::zeros<vec>((n - 1) * p);
		u = arma::zeros<vec>((n - 1) * p);
		Hb = arma::zeros<vec>((n - 1) * p);
		dl = arma::zeros<vec>(n * p);
		ddl = arma::zeros<mat>(n * p, n * p);
	}

	void b_update(double rho, arma::mat H);
	void z_update(double delta, double rho);
	void u_update();
	void Hb_update(arma::mat H);
	void l_update(vector<PPP_node> u, arma::mat pixel);
	void ADMM(double e1, double e2, double delta, double rho, arma::mat H, vector<PPP_node> u, arma::mat pixel);
};

void ADMM_iterator::l_update(vector<PPP_node> u, arma::mat pixel)
{
	vector<PPP_node>::iterator it;
	int i, j, k, i1, j1;
	double temp_intensity;
	arma::mat temp = b;

	temp.reshape(n, p);
	for (it = u.begin(); it != u.end(); it++)
	{
		i = it->index;
		i1 = floor(it->s(1) * 10);
		j1 = floor(it->s(0) * 10);
		temp_intensity = exp(sum(temp.row(i) *  it->x));
		for (j = 0; j < p; j++)
		{
			dl(j * n + i) = it->x(j) - 0.01 * it->b(j) * temp_intensity / pixel(i1, j1);
			for (k = 0; k < p; k++)
				ddl(j * n + i, k * n + i) = - 0.01 * it->x(j) * it->b(k) * temp_intensity / pixel(i1, j1);
		}
	}
}

void ADMM_iterator::b_update(double rho, arma::mat H)
{
	b = solve(rho * H.t() * H - ddl, rho * H.t() * (z - u) + dl - ddl * b);
}

void ADMM_iterator::z_update(double delta, double rho)
{
	z = soft_thresholding(delta / rho, Hb + u);
}

void ADMM_iterator::u_update()
{
	u = u + Hb - z;
}

void ADMM_iterator::Hb_update(arma::mat H)
{
	Hb = H * b;
}

void ADMM_iterator::ADMM(double e1, double e2, double delta, double rho, arma::mat H, vector<PPP_node> u, arma::mat pixel)
{
	arma::vec z_temp;
	double rsd1, rsd2;
	initial();
	do
	{
		z_temp = z;
		l_update(u, pixel);
		b_update(rho, H);
		Hb_update(H);
		z_update(delta, rho);
		u_update();
		rsd1 = abs(Hb - z).max();
		rsd2 = rho * abs(z - z_temp).max();
	}while((rsd1 > e1) || (rsd2 > e2));
		
}

int main()
{
	const int MAX_INTENSITY = 5000;
	int i, p = 2;
	int n_node = 0;
	vector<PPP_node> u;
	PPP_node v;
	//generate PPP
	for (i = 0; i < MAX_INTENSITY; i++)
	{
		v.initialize();
		if (arma::randu<vec>(1).min() < (v.intensity / MAX_INTENSITY))
		{
			v.index = n_node;
			u.push_back(v);
			n_node++;
		}
	}

	Graph E_0(0, 0), E(n_node, n_node - 1);
	arma::mat pixel, H, H_tilde;
	//MST
	pixel = pixel_generate(10, 10, u);
	E_0 = map_generate(u);
	E = E_0.MST();
	H = E.get_mat();
	H_tilde = diagblock(H, 2);
	//ADMM
	ADMM_iterator XX(n_node, 2);
	XX.ADMM(0.01, 0.01, 10000, 1000, H_tilde, u, pixel);
	//estimate error
	arma::mat beta_hat = XX.b;
	beta_hat.reshape(n_node, 2);
	arma::mat res = beta_hat;
	vector<PPP_node>::iterator it;
	for (it = u.begin(); it != u.end(); it++)
	{
		res(it->index, 0) = res(it->index, 0) - it->beta(0);
		res(it->index, 1) = res(it->index, 1) - it->beta(1);
	}
	cout << res <<endl;

//	system("pause");
	return 0;
}
