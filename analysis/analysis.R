library(limma)
#library(edgeR)
setwd('I:/R/down')
kx2=read.csv('test2.csv',row.names= 1)
library("survival")
library('ggplot2')
library('survminer')
time=kx2[1,]
status= kx2[2,]
time=as.numeric(time)
status=as.numeric(status)
y=Surv(time,status)
subc=t(as.matrix(kx2[4,]))# 
q2=survdiff(y~subc)
p.val <- 1 - pchisq(q2$chisq, length(q2$n) - 1)
p.val
kx2=as.data.frame(kx2)
fit=survfit(y~subc,data=kx2)
# http://rpkgs.datanovia.com/survminer/reference/ggsurvplot.html
ggsurvplot(fit,surv.median.line="none",pval=T,legend.title="",legend = c(0.92,0.95),
           legend.labs = c("low-risk", "high-risk"),pval.size=12, risk.table = TRUE,
           palette = c("#E7B800", "#2E9FDF"),conf.int = TRUE,
           ggtheme = theme_bw(), # Change ggplot2 theme
           font.tickslab=c(20,"bold"),font.legend= c(12,"bold"),font.x= c(20,"bold.italic"),font.y= c(20,"bold.italic"),
) 

#######################################################################
group <-kx2[4,]
group=t(group)
group[group==0]='Low'
group[group==1]='High'

design <- model.matrix(~0+factor(group))
x=kx2[-(1:4),]
colnames(design)=levels(factor(group))
rownames(design)=colnames(x)# colname
fit <- lmFit(x, design)
cont.matrix<-makeContrasts(paste0(unique(group),collapse = "-"),levels = design)
fit2=contrasts.fit(fit,cont.matrix)
fit2 <- eBayes(fit2)  

output <- topTable(fit2, n=Inf)
output2 <- topTable(fit2, p.value = 0.05, n=Inf)

mm=as.data.frame(output)
nn=as.data.frame(output2)
write.csv(mm,file = "tl.csv")
write.csv(nn,file = "t.csv")
#### ###
a<-data.frame(name=row.names(output2))
a["baseMeanLog2"]=output2["AveExpr"]
a["log2FoldChange"]=output2["logFC"]
a["padj"]=output2["adj.P.Val"]
data =a
################################DEG selection######################################
library(ggpubr)
library(ggplot2)

ggmaplot(data, main = expression("DEGs selection"), fc = 1.515,
         fdr = 0.05,size = 3,
         palette = c("#B31B21", "#1465AC", "darkgray"),
         genenames = as.vector(data$name),
         legend = "top",top = 0,label.select = c("PBK","ESCO2","CTSV"),label.rectangle = TRUE,
         font.label = c("bold", 25),
         font.legend = c("bold", 25),
         font.main = c("bold", 35),
         font.x=c("bold.italic", 20),
         font.tickslab=c(15,"bold"),
         font.y= c(20,"bold.italic"),
         ggtheme = ggplot2::theme_minimal())+theme(plot.title = element_text(hjust = 0.5))
#### ####
a = kx2[row.names(output2),]
a = rbind(kx2["class",], a)
a = a[order(a[1,])]
b=rbind(class=0,output2["logFC"])
a = cbind(b, a)
b = order(output2["logFC"])
b= b+1
b = c(1,b)
a=a[b,]
data =a
################################################################################################
library(ggplot2)
library(pheatmap)



class=data[1,-1]
class[class==0]='Low'
class[class==1]='High'
class=factor(class)
time=data[1,-1]/365
annotation_col = data.frame(Risk = class)#,  Time = t(time))
annotation_row = data.frame(
  GeneClass = factor(rep(c("Class1", "Class2"), c(201, 152)))
)
rownames(annotation_row) = rownames(data[-(1:2),])
ann_color=list(Risk=c(Low='green3',High='red2'), 
               GeneClass = c(Class1 = "#7570B3", Class2 = "#E7298A"))


data0=data[-1,]
data1<- data0[,-1]
bk = unique(c(seq(-3,3, length=50)))
bp8=pheatmap(data1,scale="row",cluster_rows = F,cluster_cols = F, show_colnames = F,
             show_rownames = F, color = colorRampPalette(c('green', 'black', 'red'))(length(bk)),
             annotation_col=annotation_col, annotation_colors=ann_color, fontsize=15,
             main="test",cex.main=1.5,angle_col = "0",annotation_names_row=F,breaks = bk)
###########################################################################################
