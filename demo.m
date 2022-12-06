clc;
close all;

%% 
addpath('data'); addpath('functions'); 
Files = dir(fullfile('data', '*.mat'));
Max_datanum = length(Files);

%% 
for data_num = 1:Max_datanum   
    Dname = Files(data_num).name;
    disp(['***********The test data name is: ***' num2str(data_num) '***'  Dname '****************'])
    load(Dname); 
    
    file_path = 'Results/';
    folder_name = Dname(1:end-4);  
    file_path_name = strcat(file_path,folder_name);
    if exist(file_path_name,'dir') == 0   
       mkdir(file_path_name);
    end
    file_mat_path = [file_path_name '/'];
    
    k_nn_num = 10;
    load lambda_pre_ladder.mat
    para_num = 350;
    flag = zeros(1,para_num);
    time = zeros(1,para_num);
    result_AMCGM = zeros(para_num,7);
 
    for i = 1:para_num 
        tic;
        lambda = lambda_pre_ladder(i);
        [predict_label,flag(i)] = AMCGM(X,Y,k_nn_num,lambda);
        time(i) = toc;
        result_AMCGM(i,:) = ClusteringMeasure(Y,predict_label);
    
        fprintf('lambda=%f\n ',lambda);
        fprintf('ACC: %f\n',result_AMCGM(i,1));
    end

    idx_right = find(flag == 1);  % finding the parameters that make the graph G have c connected components
    result_right_AMCGM = result_AMCGM(idx_right,:); 

    [max_result_ARI,idx_result_right_max] = max(result_right_AMCGM(:,7)); 
    max_result = result_right_AMCGM(idx_result_right_max,:);  
  
    idx_result_max = idx_right(idx_result_right_max);   
    opt_lambda = lambda_pre_ladder(idx_result_max);
    
    file_name = Dname;
    save ([file_mat_path,file_name],'Dname','time','result_AMCGM','result_right_AMCGM','max_result','idx_result_max','opt_lambda');

end