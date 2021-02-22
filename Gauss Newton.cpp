//Gauss Newton method for convex nonlinear least-squares problem
#include "armadillo"
#include <iostream>

using namespace std;
using namespace arma;

struct function_f	//parameters of function f_i(x)
{
	function_f *next;	//next function
	function_f *front;	//prior function
	arma::mat A;
	arma::vec b;
};

bool is_postivedefine(arma::mat A)	//identify if matrix A is positive defined 
{
	arma::mat eigen_m;
	arma::vec eigen_c;
	
	arma::eig_sym(eigen_c, eigen_m, A);
	if (eigen_c.min() > 0) return true;
	else return false;
}

arma::vec f(arma::mat A, arma::vec b, arma::vec x)	//value of each f_i(x)
{
	return 1.0 / 2 * x.t()*A*x + b.t()*x + 1.0;
}

arma::vec df(arma::mat A, arma::vec b, arma::vec x)	//derivative of each f_i(x)
{
	return A*x + b;
}

double stop_criterion(function_f *v, arma::vec x, int n)	
//stopping criterion: the maximum element of dervative of the whole function f(x) is less than e
{
	function_f *u = v;
	arma::vec a = arma::zeros<vec>(n), d_f;

	while(u != NULL)
	{
		d_f = df(u->A, u->b, x);
		a = a + d_f;
		u = u->next;
	}

	return abs(a).max();
}

arma::vec GN_step(function_f *v, arma::vec x, int n)	//step length of iteration
{
	function_f *u = v;
	arma::mat B = arma::zeros<mat>(n, n);
	arma::vec a = arma::zeros<vec>(n), d_f;

	while(u != NULL)
	{
		d_f = df(u->A, u->b, x);
		B = B + d_f*d_f.t();
		a = a + d_f*f(u->A, u->b, x);
		u = u->next;
	}

	return -solve(B, a);
}

int main()
{
	int m, n, i;
	double p, e, g;
	arma::mat B, C, A_inverse;
	arma::vec a, x_0, x_new;
	function_f *v;		//sequence of function f_i(x)
	v = new function_f;
	v->front = NULL;

	function_f *u =v;
	function_f *w;

	cout << "The number of functions\n";
	cin >> m;
	cout << "The size of A\n";
	cin >> n;
	cout << "The threhold\n";
	cin >> e;

	for (i = 1; i <= m; i++)	//randomly generate a function f(x)
	{
		do
		{
			B = arma::randu<mat>(n, n) * 1;
			C = 1.0 / 2 * (B + B.t());
		}while (!is_postivedefine(C));		
		u->A = C;		//generate matrix A which is symmetric and positive defined
		A_inverse = arma::inv_sympd(C);

		do
		{
			a = arma::randu<vec>(n);
			p = (a.t()*A_inverse*a).min();
		} while (p > 2);
		u->b = a;		//generate vector b subject to b^T*A*b < 2;

		w = new function_f;
		u->next = w;
		w->front = u;
		u = w;
	}
	u = u->front;
	delete u->next;
	u->next = NULL;

	x_0 = arma::zeros(n);
	do		//iteration of algorithm
	{
		x_new = x_0 + GN_step(v, x_0, n);
		g = stop_criterion(v, x_new, n);
		x_0 = x_new;
	} while (g > e);

	u = v;
	i = 1;
	while (u != NULL)	//print the parameter of function f(x)
	{
		cout << "Print the parameter of f_" << i <<endl;
		u->A.print("\n");
		u->b.print("\n");
		u = u->next;
		i++;
	}

	//print the result
	cout << endl;
	cout << "Print the minimum point of f(x)" << endl;
	cout << x_0 <<endl;
	
//	system("pause");
	return 0;
}



