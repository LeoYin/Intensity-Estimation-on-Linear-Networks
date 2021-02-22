#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include "mst.h"


using namespace std;
static int RANDOM_SEED;

struct PPP_node
{
	double s[2], x[2], beta[2];
	double intensity = 0;
	int index;
};

double runif()
{
	default_random_engine random(RANDOM_SEED);
	uniform_real_distribution<double> dis(0.0, 1.0);
	RANDOM_SEED = (RANDOM_SEED * 123 + 59) / 19 + 23;
	return dis(random);
}

void x_value(PPP_node *p)
{
	p->x[0] = 3.25 - abs(0.5 - p->s[0]) * 6;
	p->x[1] = 3.25 - abs(0.5 - p->s[1]) * 6;
}

void beta_value(PPP_node *p)
{
	p->beta[0] = ceil((abs(0.5 - p->s[0]) + 0.001) * 6);
	p->beta[1] = ceil((abs(0.5 - p->s[1]) + 0.001) * 6);
}

void intensity_value(PPP_node *p)
{
	int i;
	double temp = 0;
	for (i = 0; i < 2; i++)
	{
		temp += *(p->beta + i)* *(p->x + i);
	}
	p->intensity = exp(temp);
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
			r = sqrt(pow(it1->s[0] - it2->s[0], 2) + pow(it1->s[1] - it2->s[1], 2));
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

void zero_matrix(int ***a, int n_rows, int n_cols)
{
	*a = new int*[n_rows];
	for (int i = 0; i < n_rows; i++)
	{
		(*a)[i] = new int[n_cols];
		for (int j = 0; j < n_cols; j++)
			(*a)[i][j] = 0;
	}
}

void zero_vector(double **a, int n)
{
	*a = new double[n];
	for (int i = 0; i < n; i++)
		(*a)[i] = 0;
}

void pixel_generate(int **a, int m, int n, vector<PPP_node> u)
{
	int i, j;
	vector<PPP_node>::iterator it;

	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			a[i][j] = 0;
	for (it = u.begin(); it != u.end(); it++)
	{
		i = floor(it->s[1] * 10);
		j = floor(it->s[0] * 10);
		a[i][j]++;
	}                                                
}

struct sparse_matrix
{
	int n_rows, n_cols;
	vector<edge_pair> links;

	sparse_matrix(int n_rows, int n_cols)
	{
		this->n_rows = n_rows;
		this->n_cols = n_cols;
	}

	void matrix_generate(Graph g)
	{
		vector<pair<double, edge_pair>>::iterator it;
		for (it = g.edges.begin(); it != g.edges.end(); it++)
		{
			links.push_back(it->second);
		}
	}

	void inner_product(int **a)
	{
		int i, j;
		vector<edge_pair>::iterator it;
		for (it = links.begin(); it != links.end(); it++)
		{
			i = it->first;
			j = it->second;
			a[i][i]++;
			a[j][j]++;
			a[i][j]--;
			a[j][i]--;
		}
	}
};

void soft_threholding(double kappa, double *a)
{
	for (int i = 0; i < sizeof(a); i++)
		a[i] = (((1 - kappa / abs(a[i])) > 0) ? (1 - kappa / abs(a[i])) : 0)*a[i];
}

struct ADMM_iterator
{
	int n, p;
	double *x, *z, *u, *Hb;

	ADMM_iterator(int n, int p)
	{
		this->n = n;
		this->p = p;
	}

	void initial()
	{
		int i;
		x = NULL;
		z = NULL;
		u = NULL;
		Hb = NULL;
		zero_vector(&x, n*p);
		zero_vector(&z, 3 * (n - 1));
		zero_vector(&u, 3 * (n - 1));
		zero_vector(&Hb, 3 * (n - 1));
	}

	void x_update();
	void z_update(double delta, double rho, int **H);
	void u_update(int **H);
	void Hb_update(int **H);
	void ADMM(double e);
};

void ADMM_iterator::x_update()
{

}

void ADMM_iterator::Hb_update(int **H)
{
	int i, j;
	double *temp;
	temp = new double[3 * (n - 1)];
	for (i = 0; i < n - 1; i++)
	{
		temp[i] = temp[i + n - 1] = temp[i + 2 * n - 2] = 0;
		for (j = 0; j < n; j++)
		{
			temp[i] += H[i][j] * x[j];
			temp[i + n - 1] += H[i][j] * x[j + n];
			temp[i + 2 * n - 2] += H[i][j] * x[j + 2 * n];
		}
	}
	for (i = 0; i < 3 * (n - 1); i++)
		Hb[i] = temp[i];
	delete temp;
}

void ADMM_iterator::z_update(double delta, double rho, int **H)
{
	for (int i = 0; i < 3 * (n - 1); i++)
		z[i] = Hb[i] + u[i];
	soft_threholding(delta / rho, z);
}

void ADMM_iterator::u_update(int **H)
{
	for (int i = 0; i < 3 * (n - 1); i++)
		u[i] = u[i] + Hb[i] - z[i];
}

void ADMM_iterator::ADMM(double e)
{
	
}

int main()
{
	const int MAX_INTENSITY = 5000;
	int i;
	int n_node = 0;
	vector<PPP_node> u;
	PPP_node v;

	for (i = 0; i < MAX_INTENSITY; i++)
	{
		v.s[0] = runif();
		v.s[1] = runif();
		x_value(&v);
		beta_value(&v);
		intensity_value(&v);
		if (runif() < (v.intensity / MAX_INTENSITY))
		{
			v.index = n_node;
			u.push_back(v);
			n_node++;
		}
	}

	Graph E_0(0, 0), E(n_node, n_node - 1);
	sparse_matrix H(n_node - 1, n_node);
	int **pixel = NULL, **HTH = NULL;

	zero_matrix(&pixel, 10, 10);
	pixel_generate(pixel, 10, 10, u);

	E_0 = map_generate(u);
	E = E_0.MST();

	H.matrix_generate(E);
	zero_matrix(&HTH, n_node, n_node);
	H.inner_product(HTH);


	vector<edge_pair>::iterator it;
	int k = 0, t = 0;
	int s1=0, s2=0;
	for (k = 0; k < n_node; k++)
	{
		for (t = 0; t < n_node; t++)
		{
			if (k == t) s1 += HTH[k][t];
			else s2 += HTH[k][t];
		}
	}
	cout << s1 << endl << s2 << endl;
	



	system("pause");
	return 0;
}



/*
int main()
{
	vector<pair<double, int>> u;
	
	u.push_back({ 0.5,2 });
	u.push_back({ 0.7,1 });
	u.push_back({ 1.6,3 });
	sort(u.begin(), u.end());
	vector<pair<double, int>>::iterator it;
	for (it = u.begin(); it != u.end(); it++)
		cout << it->first << " " << it->second << endl;
	system("pause");
	return 0;
}
*/
/*
void out(double **a, int m, int n)
{
	int i, j;
	double b = 0.0;
	for (i = 0; i<m; i++)
	{
		for (j = 0; j<n; j++)
		{
			a[i][j] = b;
			b += 1;
			printf("%5.1f", a[i][j]);
		}
		std::cout << std::endl;
	}

}

int main()
{
	int i, j, m = 2, n = 3;
	double **a;

	a = new double*[m];
	for (i = 0; i<m; i++)
		a[i] = new double[n];

	out(a, m, n);
	cout << a[1][2];
	system("pause");
	return 0;
}
*/