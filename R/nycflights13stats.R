library(nycflights13)
library(tidyverse)
library(lm.beta)
head(flights)
 
flights_upper <- quantile(flights$dep_delay, 0.997, na.rm = TRUE)
flights_lower<- quantile(flights$dep_delay, 0.003, na.rm = TRUE)
flights_out <- which(flights$dep_delay > flights_upper | flights$dep_delay < flights_lower)
((nrow(flights) - length(flights_out))/nrow(flights)) * 100



flights_noout<- flights[-flights_out,]

Q2 <- cor.test(flights_noout$dep_delay, flights_noout$distance)
print(Q2)



Q3 <- lm(flights_noout$dep_delay ~ distance, data = flights_noout)
summary(Q3)



Q4<- lm.beta(Q3)
print(Q4)




Q5<- lm(flights_noout$dep_delay ~ flights_noout$distance + flights_noout$carrier, data = flights_noout)
summary(Q5)
