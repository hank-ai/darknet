#include "darknet_internal.hpp"

float Darknet::get_hierarchy_probability(float *x, Darknet::Tree *hier, int c)
{
	TAT(TATPARMS);

	float p = 1;
	while(c >= 0)
	{
		p = p * x[c];
		c = hier->parent[c];
	}

	return p;
}

void Darknet::hierarchy_predictions(float *predictions, int n, Darknet::Tree *hier, int only_leaves)
{
	TAT(TATPARMS);

	for (int j = 0; j < n; ++j)
	{
		int parent = hier->parent[j];
		if (parent >= 0)
		{
			predictions[j] *= predictions[parent];
		}
	}

	if (only_leaves)
	{
		for (int j = 0; j < n; ++j)
		{
			if (!hier->leaf[j])
			{
				predictions[j] = 0;
			}
		}
	}
}

int Darknet::hierarchy_top_prediction(float *predictions, Darknet::Tree *hier, float thresh, int stride)
{
	TAT(TATPARMS);

	float p = 1;
	int group = 0;
	int i;
	while (1) {
		float max = 0;
		int max_i = 0;

		for (i = 0; i < hier->group_size[group]; ++i) {
			int index = i + hier->group_offset[group];
			float val = predictions[(i + hier->group_offset[group])*stride];
			if (val > max) {
				max_i = index;
				max = val;
			}
		}
		if (p*max > thresh) {
			p = p*max;
			group = hier->child[max_i];
			if (hier->child[max_i] < 0) return max_i;
		}
		else if (group == 0) {
			return max_i;
		}
		else {
			return hier->parent[hier->group_offset[group]];
		}
	}
	return 0;
}

Darknet::Tree * Darknet::read_tree(const char *filename)
{
	TAT(TATPARMS);

	Darknet::Tree t = {0};
	FILE *fp = fopen(filename, "r");

	char *line;
	int last_parent = -1;
	int group_size = 0;
	int groups = 0;
	int n = 0;
	while((line=fgetl(fp)) != 0){
		char* id = (char*)xcalloc(256, sizeof(char));
		int parent = -1;
		sscanf(line, "%s %d", id, &parent);
		t.parent = (int*)xrealloc(t.parent, (n + 1) * sizeof(int));
		t.parent[n] = parent;

		t.name = (char**)xrealloc(t.name, (n + 1) * sizeof(char*));
		t.name[n] = id;
		if(parent != last_parent){
			++groups;
			t.group_offset = (int*)xrealloc(t.group_offset, groups * sizeof(int));
			t.group_offset[groups - 1] = n - group_size;
			t.group_size = (int*)xrealloc(t.group_size, groups * sizeof(int));
			t.group_size[groups - 1] = group_size;
			group_size = 0;
			last_parent = parent;
		}
		t.group = (int*)xrealloc(t.group, (n + 1) * sizeof(int));
		t.group[n] = groups;
		++n;
		++group_size;
	}
	++groups;
	t.group_offset = (int*)xrealloc(t.group_offset, groups * sizeof(int));
	t.group_offset[groups - 1] = n - group_size;
	t.group_size = (int*)xrealloc(t.group_size, groups * sizeof(int));
	t.group_size[groups - 1] = group_size;
	t.n = n;
	t.groups = groups;
	t.leaf = (int*)xcalloc(n, sizeof(int));
	int i;
	for(i = 0; i < n; ++i) t.leaf[i] = 1;
	for(i = 0; i < n; ++i) if(t.parent[i] >= 0) t.leaf[t.parent[i]] = 0;

	fclose(fp);
	Darknet::Tree* tree_ptr = (Darknet::Tree*)xcalloc(1, sizeof(Darknet::Tree));
	*tree_ptr = t;
	//error(0);
	return tree_ptr;
}
