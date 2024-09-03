# Heterogeneous Treatment Effects of Spinal Fusion Surgery for Adolescent Idiopathic Scoliosis Patients

Adolescent Idiopathic Scoliosis (AIS) is a prevalent spinal deformity that affects a significant proportion of adolescents worldwide. Surgical intervention, such as spinal fusion surgery, is a common treatment for severe cases, yet the variability in patient responses necessitates a deeper understanding of heterogeneous treatment effects (HTEs). In this study, we estimate the HTEs of surgical intervention in AIS patients using advanced causal machine learning. We use a comprehensive dataset comprising demographic and radiographic parameters, as well as patient-reported outcomes (PROs), to identify factors influencing treatment effects. Our approach employs two primary paradigms: (1) the counterfactual prediction (T-learner) and (2) the direct effect estimation (X-learner), both within the potential outcomes framework. We first perform balance checks analyses to ensure the comparability of treated and control groups, using standardized mean differences (SMD) and Kolmogorov–Smirnov test to assess covariate balance. We then conduct a series of regression analyses to investigate the influence of individual covariates on the estimated treatment effects. Estimation of HTEs can inform personalized treatment strategies, potentially improving patient satisfaction and clinical outcomes. This study can potentially facilitate more effective AIS patient care by estimating HTEs through causal inference in surgical treatment, providing a robust framework for personalized medicine in scoliosis care.
![Alt text](/HTE.png)
Overview of the T-Learner and X-Learner models. The T-learner trains separate models on the treated and control groups to predict the potential outcomes under each condition. At inference, the CATE is the difference in predictions. The goal of the X-Learner is for each treatment group to learn more about the other treatment group by imputing treatment effects.
