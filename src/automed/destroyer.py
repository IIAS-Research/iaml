# class Destroyer():
#     #   - Soft Destroy -> Check if it works
#     def __init__(self, patience=3, bad_result_diff=0.02, soft_destroy=True):
#         self.root_id = -1
#         self.patience = patience
#         self.children_results = {}
#         self.children_outputs_id = {}
#         self.destroyed_child = []
#         self.bad_result_diff = bad_result_diff
#         self.best_result = -1
#         self.soft_destroy = soft_destroy
        
#     def set_root(self, root_step):
#         self.root_id = id(root_step)
    
        
#     def destroyed(self, step):
#         parents = self.__parents_ids(step)
#         if self.soft_destroy:
#             child_id = parents[-1]
#         else:
#             child_id = parents[0]
            
#         if child_id in self.destroyed_child: # Already destroyed
#             return True
#         elif child_id in self.children_results.keys(): # Cannot be destroy unknown step
#             # Enough results to be destroyed
#             if len(self.children_results[child_id]) >= self.patience: 
#                 maxi = self.children_results[child_id]
#                 if maxi < self.best_result * (1 - self.bad_result_diff): # Bad results
#                     # DESTROY !
#                     self.destroyed_child.append(child_id)
#                     return True
                    
#         return False # Everything is fine, not destroyed
            
    
#     def track_output(self, step, output):
#         parents = self.__parents_ids(step)
#         if self.soft_destroy:
#             child_id = parents[-1]
#         else:
#             child_id = parents[0]
            
#         if child_id not in self.children_results.keys():
#             self.children_results[child_id] = []
            
#         if child_id not in self.children_outputs_id.keys():
#             self.children_outputs_id[child_id] = []
        
#         if id(output) not in self.children_outputs_id[child_id]:
#             self.children_results[child_id].append(output.computed_metrics)
#             self.children_outputs_id[child_id].append(id(output))
        
#             if output.computed_metrics > self.best_result:
#                 self.best_result = output.computed_metrics
            
    
#     # def __find_first_child_id(self, step):
#     #     id_list = step.parents_steps + [id(step)]
#     #     root_index = id_list.index(self.root_id)
#     #     return id_list[root_index+1]
    
    
#     def __parents_ids(self, step):
#         id_list = step.parents_steps + [id(step)]
#         root_index = id_list.index(self.root_id)
#         return id_list[root_index+1:]
