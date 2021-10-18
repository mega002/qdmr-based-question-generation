from qdmr_transforms.src.qdmr_transforms import QDMRTransform

if __name__ == '__main__':
    qdmr = "return bed ;return canopy of #1 ;return color of #2 ;return if  #3 is aqua ;return four posts ;return if  #2 is gathered at #5 ;return if  both  #4 and #6"
    qdmr = "return store fronts ;return awning of #1 ;return #1 where  #2 is brown ;return number of  #3 ;return if  #4 is equal to  one"
    qdmr = "return papers ;return #1 from  PVLDB ;return #2 before 2005 ;return citations of  #3 ;return number of  #4 for each  #3 ;return sum of  #5"
    qdmr = "return flights ;return #1 from  boston to san francisco ;return #2 after 8am ;return the first #3"
    qdmr = "return objects ;return #1 that  are purple ;return #2 that  are metallic ;return the  largest #3 ;return the  shape of #4"
    qdmr = "return China launched four invasions of Burma ;return years of #1 ;return the  highest of #2 ;return the  lowest of #2 ;return the  difference of #3 and  #4"
    qdmr = "return objects ;return #1 that are purple ;return #1 that are red ;return number of  #2 ;return number of  #3 ;return which  is  more of #4 , #5"
    qdmr = "return the  large red sphere ;return if #1 is rubber ;return if #1 is metallic ;return which is true of #2 ,  #3"
    qdmr = "return objects  ;return #1 that are green ;return #1 that are cylinders ;return number of #2 ;return number of #3 ;return which is lowest of  #4 ,   #5"
    qdmr = "return Mind Dynasty conquering Yunnan borderlands ;return Burmese control of the  Shan States ;return when  was  #1 ;return when  was  #2 ;return which is lowest of #3 ,  #4"
    qdmr = "return conferences ;return papers of #1 ;return #2 that  contain the keyword Relational Database ;return number of #3 for each #1 ;return #1 where #4 is  more than  60"
    qdmr = "return flights ;return #1 that  are round trip ;return #2 from  los angeles ;return #3 to tacoma washington ;return #4 with  a  stopover in san francisco ;return #5 on  june tenth 1993 ;return prices of #6 ;return #6 where #7 is  at most 300 dollars"
    qdmr = "return Charlie Hunnam ;return movies of #1 ;return casting directors of #2 ;return #2 where #3 is Margery Simkin"
    qdmr = "return US State ;return total population of #1 ;return #1 where #2 is 1005163 ;return #3 that the  Missouri river bisects"
    qdmr = "return images ;return hounds in  #1 ;return #2 that are standing ;return number of  #3 for each  #1 ;return #1 where  #4 is equal to  one ;return number of  #5 ;return number of  #1 ;return if  #6 is equal to  #7"
    qdmr = "return products ;return stock number of  #1 ;return #1 where  #2 starts with  2 ;return names of  #3"
    qdmr = "return competitions ;return type of  #1 ;return #1 where #2 is  not  tournament ;return different years for #3"
    qdmr = "return departures ;return #1 from  boston  ;return #2 to atlanta   ;return when  is #3 ;return #3 where #4 is lowest ;return return trips ;return #6 from  atlanta   ;return #7 to boston  ;return when  is #8 ;return #8 where #9 is highest ;return #5 ,  #10"
    qdmr = "return objects ;return #1 that is shiny ;return closeness of #2 ;return #2 where #3 is highest ;return shape of #4"
    qdmr = "return the  Manipuri chief ;return did #1 present a son to Ava ;return did #1 present a daughter to Ava ;return which is true of #2 ,  #3"
    qdmr = "return objects ;return #1 that are shiny ;return #2 that are green ;return if #3 is large ;return if #3 is a cube ;return if both #4 and  #5 are true"
    qdmr = "return the  venue in  Sydney ;return the  capacity of #1 ;return #1 where #2 is 10,000 people"

    example = QDMRExample("111", "????", qdmr)
    transform = OpReplaceTransform(example)
    print(f"* Original qdmr:\n{qdmr}\n")
    print("* Num of transformations: ", len(transform.transformations()))
    for t in transform.transformations():
        print(t.qdmr)
        print(t.transform)
        print("*" * 20)


