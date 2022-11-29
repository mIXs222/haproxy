/* This header can be read by both C and C++ compilers */
#ifndef MLLB_H
#define MLLB_H
#ifdef __cplusplus
extern "C" {
#endif
void* ModelLB_new(char param_path[]);
void ModelLB_delete(void* mlb_void);
int ModelLB_server_up(void* mlb_void, void* server_struct);
void ModelLB_server_down(void* mlb_void, void* server_struct, int server_id);
void ModelLB_take_conn(void* mlb_void, void* server_struct, int server_id);
void ModelLB_drop_conn(void* mlb_void, void* server_struct, int server_id);
void ModelLB_update_state(void* mlb_void);
void* ModelLB_select_server(void* mlb_void);
#ifdef __cplusplus
}  // extern "C"
#endif
#endif /*MLLB_H*/