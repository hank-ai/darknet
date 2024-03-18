#include "option_list.hpp"
#include "darknet_internal.hpp"

list *make_list()
{
	TAT(TATPARMS);

	list* l = (list*)xmalloc(sizeof(list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}

void *list_pop(list *l)
{
	TAT(TATPARMS);

	if(!l->back) return 0;
	node *b = l->back;
	void *val = b->val;
	l->back = b->prev;
	if(l->back) l->back->next = 0;
	free(b);
	--l->size;

	return val;
}

void list_insert(list *l, void *val)
{
	TAT(TATPARMS);

	node* newnode = (node*)xmalloc(sizeof(node));
	newnode->val = val;
	newnode->next = 0;

	if(!l->back){
		l->front = newnode;
		newnode->prev = 0;
	}else{
		l->back->next = newnode;
		newnode->prev = l->back;
	}
	l->back = newnode;
	++l->size;
}

void free_node(node *n)
{
	TAT(TATPARMS);

	node *next;
	while(n) {
		next = n->next;
		free(n);
		n = next;
	}
}

void free_list_val(list *l)
{
	TAT(TATPARMS);

	node *n = l->front;
	node *next;
	while (n) {
		next = n->next;
		free(n->val);
		n = next;
	}
}

void free_list(list *l)
{
	TAT(TATPARMS);

	free_node(l->front);
	free(l);
}

void free_list_contents(list *l)
{
	TAT(TATPARMS);

	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}

void free_list_contents_kvp(list *l)
{
	TAT(TATPARMS);

	node *n = l->front;
	while (n) {
		kvp* p = (kvp*)n->val;
		free(p->key);
		free(n->val);
		n = n->next;
	}
}

void **list_to_array(list *l)
{
	TAT(TATPARMS);

	void** a = (void**)xcalloc(l->size, sizeof(void*));
	int count = 0;
	node *n = l->front;
	while(n){
		a[count++] = n->val;
		n = n->next;
	}
	return a;
}
