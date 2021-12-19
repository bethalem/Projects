library(openintro)
library(tidyverse)
library(lm.beta)
fastfood <- openintro::fastfood

Q1 <- cor(na.omit(fastfood[fastfood$restaurant %in% c('Sonic', 'Subway','Taco Bell'),c('calories','total_fat','sugar','calcium')]))
round(Q1, 2)

data1 = filter(fastfood,restaurant =='Mcdonalds' | restaurant == 'Subway')
data1$restaurant = ifelse(data1$restaurant == 'Mcdonalds',1,0)
food_model = glm(formula = restaurant~calories+sodium+protein, data1,family=binomial())
food_model2 <- print(food_model)$coefficients
round(food_model2, 2)


food_model3 <- print(food_model)$aic
round(food_model3,2)


food_model1 = glm(formula = restaurant~calories+protein, data1,family=binomial())
summary(food_model1)
AIC(food_model1)


food_2 = lm(formula = calories~sat_fat +fiber+sugar, fastfood)
print(round(food_2$coefficients,2))[2]

Q4 <- lm.beta(food_2)
print(Q4)


ff <- group_by(fastfood, restaurant)
bb <- filter(ff, n() > 50 & n() < 60)
food_5 = lm(formula = total_fat~cholesterol+total_carb+vit_a+restaurant, bb)
tt<-lm.beta(food_5)
summary(tt)

ff <- group_by(fastfood, restaurant)
bb <- filter(ff, n() > 50 & n() < 60)
food_6 = lm(formula = total_fat~cholesterol+total_carb+restaurant, bb)
zz<-lm.beta(food_6)
print(round(zz$standardized.coefficients,2))[2]




