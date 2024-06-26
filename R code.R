library(tidyverse)
library(caret)
library(party)
library(DataExplorer)
library("caretEnsemble")
library(pROC)

dat<-read.csv("2307dat_mod.csv")
dat = subset(dat, select=c(TP,Age,Male,Surgery_time,Surgery,hormone_day1,
                           antiplatelet_day1,shock_day1,SOFA,Sepsis,
                           Hypertension,Diabete,
                           WBC,plt_icu_fri,Cr_icu_fri,Hemoglobin,
                           ALTicu,ASTicu,PT,APTT,
                           sysbp_max,sysbp_min,heartrate_max,heartrate_min,
                           tempc_max,tempc_min,
                           Hospital_mortality,ICUstay,HXJstay,Hosp..LOS,Cost))
library(CBCgrps)
tab1<-twogrps(dat,gvar = "TP",norm.rd = 1,cat.rd = 1,
              skewvar = c("HXJstay","ICUstay","Hosp..LOS","Cost",
                          "ALTicu","ASTicu",
                          "PT") )
#202404
dat= subset(dat, select=c(TP,gcs,Dopamine,Dobutamine,Norepinephrine,Adrenaline,
                         aspirin_day1,tirofiban_day1,P2Y12_inhibitors_day1,
                         short_acting_glucocorticoids_day1,
                         intermediate_glucocorticoids_day1,
                         long_acting_glucocorticoids_day1,Bilirubin_Total,Po2,FIO2,Bicarbonate,Sodium,Potassium,Calcium,Magnesium))

dat= subset(dat, select=c(TP,Crp,PCT,Bilirubin_Total,Po2,FIO2,Bicarbonate,Sodium,Potassium,Calcium,Magnesium))
dat=dat%>% 
  mutate(PF=(Po2/FIO2)*100) 
skewvar2 = c( "Bilirubin_Total" ,"PCT" ,"Crp"  )
tab1<-twogrps(dat,gvar = "TP",norm.rd = 1,cat.rd = 1,sk.rd=1,minfactorlevels = 5,
              skewvar = skewvar2 )

dat = subset(dat, select=c(class,Age,Male,hormone_day1,
                           antiplatelet_day1,shock_day1,SOFA,Sepsis,
                           Hypertension,Diabete,
                           WBC,plt_icu_fri,Cr_icu_fri,Hemoglobin,
                           ALTicu,ASTicu,PT,APTT,
                           sysbp_max,sysbp_min,heartrate_max,heartrate_min,
                           tempc_max,tempc_min,
                           Hospital_mortality,ICUstay,HXJstay,Hosp..LOS,Cost))
library(CBCgrps)
tab1<-multigrps(dat,gvar = "class",norm.rd = 1,cat.rd = 1, sk.rd=1,minfactorlevels = 6,
                skewvar = c("HXJstay","ICUstay","Hosp..LOS","Cost",
                            "ALTicu","ASTicu",
                            "PT"))
colnames(tab1) <- c("Col1", "Col2", "Col3", "Col4", "Col5","Col6","Col7")
flextable::save_as_docx(flextable::flextable(as.data.frame(tab1)),
                        path = "2024tab1.docx")

print(tab1,quote = TRUE)  

#randomForest
dat=read.csv("2404dat_mod.csv")
mdrrClass=dat$TP
dat=subset(dat, select=-c(TP))
subsets = c(10,20,30,40,50,60,70)       
ctrl= rfeControl(functions = rfFuncs, method = "cv",verbose = FALSE, returnResamp = "final")
Profile = rfe(dat, mdrrClass, sizes = subsets, rfeControl = ctrl)	
library(randomForest)
print(Profile)
plot(Profile)
Profile$optVariables 

#lasso
library(glmnet)
library(ggplot2)
X <- as.matrix(dat[, 1:(ncol(dat) - 1)])   
y <- as.vector(dat[, ncol(dat)])     
 
cv_lasso <- cv.glmnet(X, y, alpha = 1)
best_lambda <- cv_lasso$lambda.min*7
 
final_lasso_model <- glmnet(X, y, alpha = 1, lambda = best_lambda)
coefficients <- coef(final_lasso_model)
 
print(coefficients[coefficients != 0])
lasso_path <- glmnet(X, y, alpha = 1)
 
plot(lasso_path, xvar = "lambda", label = TRUE, main = "Lasso Path")
abline(v = log(best_lambda), lty = 2, col = "red")
legend("topright", legend = c("Best Lambda"), col = "red", lty = 2)
 
cv_errors <- data.frame(
  lambda = log(cv_lasso$lambda),
  mse = cv_lasso$cvm,
  lower_ci = cv_lasso$cvlo,
  upper_ci = cv_lasso$cvup
)
 
g <- ggplot(cv_errors, aes(x = lambda, y = mse)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.1) +
  geom_vline(xintercept = log(best_lambda), linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(x = "log(Lambda)", y = "Cross-Validation Error", title = "Cross-Validation Error vs. Lambda") +
  annotate("text", x = log(best_lambda) + 0.5, y = max(cv_errors$mse), label = "Best Lambda", color = "red")
print(g)
 
non_zero_indices <- which(coefficients != 0)
 
selected_variables <- rownames(coefficients)[non_zero_indices]
 
print(selected_variables)
 
dat=dat[,non_zero_indices]
non_zero_indices


dat<-read.csv("2307dat_mod.csv")
dat=dat[dat$Surgery_time==1,]

 
dat= subset(dat, select=c(plt_icu_fri,Hemoglobin,ASTicu,sysbp_min,sysbp_max,heartrate_mean,
                          heartrate_max,Cr_icu_fri,TP)) 
colnames(dat)=c("Platelet","Hemoglobin","Ast","Sysbp_min","Sysbp_max",         
                "Heartrate_mean","Heartrate_max","Creatinine","PostComplication")

dat$PostComplication=ifelse(dat$PostComplication==1,"Yes","No")
dat$PostComplication=as.factor(dat$PostComplication)

set.seed(3456)
trainIndex <- createDataPartition(
  y = dat$PostComplication, p = .7, 
  list = FALSE, 
  times = 1)
datTrain <- dat[trainIndex,]
datTest <- dat[-trainIndex,]

fitControl <- trainControl( method = "repeatedcv",
                            number = 10, repeats = 10)
gbmGrid <-  expand.grid(  interaction.depth = c(1, 5, 9), 
                          n.trees = (1:30)*50,  shrinkage = 0.1, n.minobsinnode = 20)
gbmFit <- caret::train(  PostComplication ~ ., 
                         data = datTrain,   method = "gbm",   trControl = fitControl,
                         tuneGrid = gbmGrid,  verbose = FALSE)
summary(gbmFit)

train_control <- trainControl(  method="boot",
                                number=25,  savePredictions="final",  classProbs=TRUE,
                                index=createResample(datTrain$PostComplication, 25),
                                summaryFunction=twoClassSummary)
model_list <- caretList(  
  PostComplication~., data=datTrain,
  trControl=train_control,  metric="ROC",
  tuneList=list( 
    SVM=caretModelSpec( method="svmLinearWeights", 
                        tuneGrid=expand.grid(  cost=seq(0.1,1,0.2),  weight=c(0.5,0.8,1))),
    C5.0=caretModelSpec(  method="C5.0", 
                          tuneGrid=expand.grid( trials=(1:5)*10,
                                                model=c("tree", "rules"),  winnow=c(TRUE, FALSE))),
    Bayes=caretModelSpec( method="naive_bayes"),
    XGboost=caretModelSpec( method="xgbTree",  tuneGrid=expand.grid(
      nrounds=(1:5)*10,  max_depth= 6, eta=c(0.1),gamma= c(0.1),
      colsample_bytree=1,  min_child_weight=c(0.5,0.8,1),
      subsample=c(0.3,0.5,0.8))) ))
gbm_ensemble <- caretStack(
  model_list,  method="gbm",  verbose=FALSE,
  tuneLength=10,  metric="ROC",  trControl=trainControl(
    method="boot",   number=10,   savePredictions="final",
    classProbs=TRUE, summaryFunction=twoClassSummary  ))
#####分开
plot(model_list$C5.0)
plot(model_list$SVM)
plot(model_list$XGboost)
library("PerformanceAnalytics")
dtResample <- resamples(model_list)$values %>% 
  dplyr::select(ends_with("~ROC")) %>% 
  rename_with(~str_replace(.x,"~ROC","")) %>% 
  chart.Correlation() 
library(cowplot)
model_preds <- lapply(
  model_list, predict, 
  newdata=datTest, type="prob")
model_preds <- lapply(
  model_preds, function(x) x[,"Yes"])
model_preds <- data.frame(model_preds)
model_preds$Ensemble <- 1-predict(
  gbm_ensemble, newdata=datTest,
  type="prob")
model_roc <- lapply(
  model_preds, function(xx){
    roc(response=datTest$PostComplication,
        direction = "<",predictor = xx)
  })


#Add model performance of the ensemble model
model_roc$Ensemble <- roc(
  response=datTest$PostComplication,
  direction = "<",
  predictor = model_preds$Ensemble)
model_TextAUC <- lapply(model_roc, function(xx){
  paste("AUC: ",round(pROC::auc(xx),3),
        "[",round(pROC::ci.auc(xx)[1],3),",",
        round(pROC::ci.auc(xx)[3],3),"]",sep = "")
})

names(model_roc) <- paste(names(model_TextAUC),unlist(model_TextAUC))
plotROC <- ggroc(model_roc)+
  theme(legend.position=c(0.6,0.3))+
  guides(color=guide_legend(title="Models and AUCs"))
datCalib <- cbind(model_preds,testSetY=datTest$PostComplication)
#datCalib
#write.csv(datCalib,file = "datCalib_mimic.csv")

cal_obj <- calibration(relevel(testSetY,ref = "Yes")  ~ SVM +
                         C5.0+XGboost+Bayes+Ensemble,
                       data = datCalib,
                       cuts = 6)

#+Ensemble
calplot <- plot(cal_obj, type = "b", auto.key = list(columns = 3,
                                                     lines = TRUE, points = T),xlab="Predicted Event Percentage")

#
ggdraw() +
  draw_plot(calplot, 0,0.5,  1, 0.5) +
  draw_plot(plotROC, 0, 0, 1, 0.5) +
  draw_plot_label(c("A", "B"), 
                  c(0, 0), c(1, 0.5), size = 15)
library(gbm)
ggplot(caret::varImp(gbmFit))
ggplot(caret::varImp(model_list$C5.0))
ggplot(caret::varImp(model_list$SVM))
ggplot(caret::varImp(model_list$Bayes))

##C5.0
library(lime)
explanation<-lime(datTrain,model_list$C5.0)
exp<-lime::explain(
  datTrain[1:2,], 
  explanation,n_labels = 2,n_features = 10)
plot_explanations(exp)
plot_features(exp, ncol = 2)
##模型C5.0
library("iBreakDown")
library("DALEX")
p_fun <- function(object, newdata){
  1-predict(object, newdata = newdata, 
            type = "prob") }
Ensmeble_la_un <- break_down_uncertainty(
  model_list$C5.0, 
  data = datTrain[,!names(datTrain)%in%"PostComplication"],
  new_observation = datTrain[1,],
  predict_function = p_fun,
  path = "average")
plot(Ensmeble_la_un)
#class
train_control <- trainControl(  method="boot",
                                number=25,  savePredictions="final",  classProbs=TRUE,
                                index=createResample(datTrain$PostComplication, 25),
                                summaryFunction=twoClassSummary)

SVMFit= caret::train(  PostComplication ~ ., data = datTrain,   method="svmLinearWeights",
                       trControl = fitControl,
                    tuneGrid=expand.grid(  cost=seq(0.1,1,0.2),  weight=c(0.5,0.8,1)),verbose = FALSE)

rfFit <- caret::train(PostComplication ~ ., data = datTrain, method = "rf", trControl = fitControl, verbose = FALSE)               
                    
C5.0=caret::train(  PostComplication ~ ., data = datTrain,  method="C5.0", 
                      tuneGrid=expand.grid( trials=(1:5)*10,   model=c("tree", "rules"),  winnow=c(TRUE, FALSE)))  

preds <- predict(C5.0, datTest)
confMat <- confusionMatrix(preds, datTest$PostComplication)
print(confMat) 

XGboost=caret::train(  PostComplication ~ ., data = datTrain,  method="xgbTree",  tuneGrid=expand.grid(
  nrounds=(1:5)*10,  max_depth= 6, eta=c(0.1),gamma= c(0.1),
  colsample_bytree=1,  min_child_weight=c(0.5,0.8,1),
  subsample=c(0.3,0.5,0.8)))  
##mimic
dat<-read.csv("C:/Users/lxqji/OneDrive/R/23高龄血小板减少/数据处理/mimic_com.csv")

library(CBCgrps)
tab1<-twogrps(dat,gvar = "TP",norm.rd = 1,cat.rd = 1,
              skewvar = c( "iculos", "hxjstay","hos_los","scr_in",
                           "urea_initial", "pt" ,"gas_lactate"  ) )
print(tab1,quote = TRUE)

mimic_test = subset(df, 
                    select=c(plt_initial,HB,ast,sysbp_min,
                             sysbp_max,heartrate_mean,heartrate_max,scr_in,TP)) 
colnames(mimic_test)=c("Platelet","Hemoglobin","Ast","Sysbp_min","Sysbp_max",         
                       "Heartrate_mean","Heartrate_max","Creatinine","PostComplication")
mimic_test$Hemoglobin=mimic_test$Hemoglobin*10
write.csv(mimic_test,file = "mimic_test2308.csv")
#mimic  验证
mimic_test=read.csv("C:/Users/lxqji/OneDrive/R/23高龄TP机器学习/数据处理/mimic_test2308.csv")
mimic_test$PostComplication=as.factor(mimic_test$PostComplication)
datTest=mimic_test


library(MLmetrics)
library(pander)
library(officer)

model_names <- c("SVM", "C5.0", "Bayes", "XGboost", "Ensemble")
results <- list()
for(model in model_names) {
  pred_prob <- data[[model]]
  pred <- ifelse(pred_prob > 0.5, "Yes", "No") # 你可以根据具体情况调整阈值
  obs <- data$testSetY
  acc <- Accuracy(pred, obs) # 计算Accuracy
  prec <- Precision(pred, obs) # 计算Precision
  rec <- Recall(pred, obs) # 计算Recall
  spec <- Specificity(pred, obs) # 计算Specificity
  bal_acc <- (rec + spec)/2 # 计算Balanced Accuracy
  results[[model]] <- round(c(acc, prec, rec, spec, bal_acc), 3)
}

# 将列表转换为数据框并添加行名
results_df <- data.frame(do.call(rbind, results))
row.names(results_df) <- model_names

# 修改列名
colnames(results_df) <- c("Accuracy", "Precision", "Recall", "Specificity", "Balanced Accuracy")

# 显示结果
print(results_df)

# 将数据框转为强制的数据框形式，确保其能添加到word文档，添加模型名称到第一列
results_df <- data.frame(lapply(results_df, as.character), stringsAsFactors=FALSE)
results_df <- data.frame(Model = rownames(results_df), results_df, row.names = NULL)

# 创建word文档并添加表格
doc <- read_docx() 
doc <- body_add_table(doc, results_df)

# 保存结果到word文件
print(doc, target = "混淆矩阵1.docx")

#fig 2
  library(ggplot2)

variables <- c('ICU length of stay(days)', 'Ventilation duration(days)','Length of hospital stay(days)', 'Cost(×10^4 yuan for Cost)')
NoThrombocytopenia <- c(2.7, 0.7, 20, 5.16)
MildThrombocytopenia <- c(5.8, 1.7, 25, 8.76)
ModerateThrombocytopenia <- c(9.1, 3.7, 28, 9.99)
SevereThrombocytopenia <- c(7.2, 6.3, 15, 6.52)
data <- data.frame(variables, NoThrombocytopenia, MildThrombocytopenia, ModerateThrombocytopenia, SevereThrombocytopenia)

data_melt <- reshape2::melt(data, id.vars = 'variables')

p <- ggplot(data_melt, aes(x = variables, y = value, fill = variable)) +
  geom_bar(stat = 'identity', position = position_dodge()) +
  theme_bw() +
  ylab('Value') +
  scale_fill_discrete(name = "Thrombocytopenia Levels", breaks = c("NoThrombocytopenia", "MildThrombocytopenia", "ModerateThrombocytopenia", "SevereThrombocytopenia"), labels = c("No Thrombocytopenia", "Mild Thrombocytopenia", "Moderate Thrombocytopenia", "Severe Thrombocytopenia")) + 
  ggtitle('Comparison of Outcomes Across Different Degrees of Thrombocytopenia')

p + annotate("text", x = 1, y = 29, label = "P < 0.001")+
  annotate("text", x = 2, y = 29, label = "P < 0.001")+
  annotate("text", x = 3, y = 29, label = "P < 0.001")+
  annotate("text", x = 4, y = 29, label = "P < 0.001")+
  theme(legend.position = c(.85, .7))






