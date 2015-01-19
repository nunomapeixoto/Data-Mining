library(bitops)
library(ggplot2)
library(FSelector)
library(knitr)
library(pander)
library(car)
library(rpart.plot)
library(DMwR)
library(caret)
library(performanceEstimation)
library(e1071)
library(kernlab)
library(bibtex)
require(knitcitations)
library(nnet)
library(cluster)
library(ipred)
library(kknn)
library(packrat)

out_tax <- function(data, att){
  vector = c()
  i <- 0
  for(k in att){
    i <- i+1
    num_out <- length(boxplot.stats(data[,k])$out)
    perc_out <- round(num_out/nrow(data)*100, 2)
    res <- paste(num_out, " (", paste(perc_out,"%",sep=""), ")", sep="")
    vector[i] = res
  }
  return(vector)
}

stand_dev <- function(data, att){
  vector = c()
  i <- 0
  for(k in att){
    i <- i+1
    vector[i] = sd(data[,k])
  }
  return(vector)
}

summaryToDataFrame <- function(data_sm){
  vals <- c()
  len = length(data_sm)
  x=1
  for(i in 1:len){
    tmp <- strsplit(data_sm[i],":")
    c <- tmp[[1]]
    label <- trim.spaces(c[1])
    value <- trim.spaces(c[2])
    vals <- append(vals,label,x-1)
    vals[length(vals)+1] <- value
    x <- x+1
  }
  m <- matrix(vals, nrow=2,ncol=len,byrow=TRUE)
  return(m)
}

getWF <- function(model){
  x=1
  parameters <- c()
  wf <- topPerformers(model)$new_train_data[1]$Workflow
  pars <- getWorkflow(wf,model)@pars[[1]]
  
  len_pars <- length(pars)
  for(k in 1:len_pars){
    par_value <- pars[k][[1]]
    par_name <- paste("**", names(pars[k]), "**", sep="")
    
    parameters <- append(parameters,par_name,x-1)
    parameters[length(parameters)+1] <- par_value
    x <- x+1
  }
  
  m <- matrix(parameters, nrow=2,ncol=len_pars,byrow=TRUE)
  return(m)
}