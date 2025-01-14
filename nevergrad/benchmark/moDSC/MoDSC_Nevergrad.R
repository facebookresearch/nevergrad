library(sendmailR)
library(httr)
library(RJSONIO)
library(RCurl)
httr::set_config( config( ssl_verifypeer = 0L ) )

#Set your username and password before you continue
#username<-""
#password<-""
url_temp<-"https://ws.ijs.si:8443/dsc-1.5/service/"


split_per_problem<-function(data,problem){
	data_temp<-data[which(as.character(data$problem)==problem),]
	return(data_temp)
}

split_per_algorithm<-function(data_temp,algorithm,DIM){
	data_alg<-data_temp[which(as.character(data_temp$algorithm)==algorithm),]
	data_alg<-data_alg[, seq(4,(5+DIM-1),1)]
	return(data_alg)
}

Registration_json<-function(name,affiliation,email,username,password){
	return(list(name=name,affiliation=affiliation,email=email,username=username,password=password))
}

Registration_client<-function(postfield){
	url<-paste(url_temp,"manage/user",sep="",collapse="")
	result_json<-POST(url,add_headers(.headers = c('Content-Type'="application/json",'Accept' = "application/json")),body=postfield,encode="json")
	if(result_json$status_code==200){
		return("Success!")
	}
	else{
		return("Incorrect JSON!")
	}

}

DSC_rank_service_parser<-function(name,alpha,epsilon,monte_carlo_iterations,data_raw){
	DIM<-ncol(data_raw)-4
	algorithms_raw<-unique(as.character(data_raw$algorithm))
	problems_raw<-unique(as.character(data_raw$problem))

	l<-list()
	for(i in 1:length(algorithms_raw)){
		problems<-list()
		for(j in 1:length(problems_raw)){
			x<-split_per_problem(data_raw,problems_raw[j])	
			y<-split_per_algorithm(x,algorithms_raw[i],DIM)
			y<-as.character(y$y)
			problems[[j]]<-list(name=problems_raw[j],data=y)
		}
	l[[i]]<-list(algorithm=algorithms_raw[[i]],problems=problems)
	
	}

	method<-list(name=name,alpha=alpha)

	json_temp<-list(epsilon=epsilon,monte_carlo_iterations=monte_carlo_iterations,method=method,data=l)
	return(json_temp)

}


DSC_client<-function(username,password,postfield){
	url<-paste(url_temp,"rank",sep="",collapse="")
	result_json<-POST(url,authenticate(username,password),add_headers(.headers = c('Content-Type'="application/json",'Accept' = "application/json")),body=postfield,encode="json")
	if(result_json$status_code==200){
		return(content(result_json)$result)
	}
	else{
		return("Incorrect JSON!")
	}
		
}



Omnibus_json<-function(result,alpha,name,parametric_tests){
	method<-list(name=name,alpha=alpha)
	number_algorithms<-length(res$ranked_matrix[[1]]$result)
	parametric_tests<-parametric_tests
	
	json_temp<-list(method=method,ranked_matrix=result$ranked_matrix,number_algorithms=number_algorithms,parametric_tests=parametric_tests)

	
}

Omnibus_client<-function(username,password,postfield){
	url<-paste(url_temp,"omnibus",sep="",collapse="")
	result_json<-POST(url,authenticate(username,password),add_headers(.headers = c('Content-Type'="application/json",'Accept' = "application/json")),body=postfield,encode="json")
	if(result_json$status_code==200){
		return(content(result_json)$result)
	}
	else{
		return("Incorrect JSON!")
	}	

}


Posthoc_json<-function(result,k,n,base_algorithm){
	return(list(algorithm_means=result$algorithm_means,method=result$method,base_algorithm=base_algorithm,k=k,n=n))
}

Posthoc_client<-function(username,password,postfield){	
	url<-paste(url_temp,"posthoc",sep="",collapse="")
	result_json<-POST(url,authenticate(username,password),add_headers(.headers = c('Content-Type'="application/json",'Accept' = "application/json")),body=postfield,encode="json")
	if(result_json$status_code==200){
		return(content(result_json)$result)
	}
	else{
		return("Incorrect JSON!")
	}	
}

Ensemble_create_json<-function(l,names,method){
	l_temp<-list()
	for(i in 1:length(l)){
		l_temp[[names[i]]]=l[[i]]$ranked_matrix
	}
	return(list(method=method,ensembles=l_temp))
	
}


Ensemble_support_json_client<-function(username,password,postfield){
	url<-paste(url_temp,"support/create/ensemble",sep="",collapse="")
	result_json<-POST(url,authenticate(username,password),add_headers(.headers = c('Content-Type'="application/json",'Accept' = "application/json")),body=postfield,encode="json")
	if(result_json$status_code==200){
		return(content(result_json)$result)
	}
	else{
		return("Incorrect JSON!")
	}	

}


Ensemble_client<-function(username,password,postfield){
	url<-paste(url_temp,"ensemble",sep="",collapse="")
	result_json<-POST(url,authenticate(username,password),add_headers(.headers = c('Content-Type'="application/json",'Accept' = "application/json")),body=postfield,encode="json")
	if(result_json$status_code==200){
		return(content(result_json)$result)
	}
	else{
		return("Incorrect JSON!")
	}	

}

moDSC_client<-function(username,password,postfield){
	url<-paste(url_temp,"multiobjective",sep="",collapse="")
	result_json<-POST(url,authenticate(username,password),add_headers(.headers = c('Content-Type'="application/json",'Accept' = "application/json")),body=postfield,encode="json")
	if(result_json$status_code==200){
		return(content(result_json)$result)
	}
	else{
		return("Incorrect JSON!")
	}	

}
