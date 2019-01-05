from tf_api.person_item_association import Person_Item_Association

person_association = Person_Item_Association()
persons_in_store={}

def add_new_person(person_id):
    person = Person_Item_Association()
    persons_in_store[person_id] = person

def get_person_object(person_id):
    if persons_in_store[person_id] is None:
        add_new_person(person_id)


    return persons_in_store[person_id]

