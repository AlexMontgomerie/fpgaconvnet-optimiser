import random


def apply_random_grouping(self):
    possible_switches=self.get_possible_switches()
    switch=possible_switches[random.randint(0,len(possible_switches)-1)]
    self.switch_group(switch["partition_id"],switch["target_group"])

def get_possible_switches(self):
    switch_pairs=list()
    for id,group in self.groups.items():
        if len(group)>0:
            if id>0:
                switch_pairs.append({"partition_id":group[0],"target_group":id-1})
            if id<len(self.groups)-1:
                switch_pairs.append({"partition_id":group[len(group)-1],"target_group":id+1})
    return switch_pairs

def switch_group(self,partition_id,target_group_id):
    group=self.groups[target_group_id]
    #print("I did something")
    if len(group)>0:
        if self.get_group_id(partition_id)>target_group_id:
            if(group[len(group)-1]==partition_id-1):
                self.groups[self.get_group_id(partition_id)].remove(partition_id)
                group.append(partition_id)
        elif self.get_group_id(partition_id)==target_group_id:
            return
        else:
            if(group[0]==partition_id+1):
                self.groups[self.get_group_id(partition_id)].remove(partition_id)
                group.insert(0,partition_id)
    else:
        self.groups[self.get_group_id(partition_id)].remove(partition_id)
        group.append(partition_id)