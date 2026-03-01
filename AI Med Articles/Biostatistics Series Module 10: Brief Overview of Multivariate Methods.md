# Biostatistics Series Module 10: Brief Overview of Multivariate Methods

Source: [Biostatistics Series Module 10: Brief Overview of Multivariate Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC5527714/)

## Abstract

Multivariate analysis refers to statistical techniques that simultaneously consider **three or more variables** for each subject, aiming to identify or clarify relationships among them.

Two broad classes:

1. **Dependence techniques**
   - Explore relationships between **one or more dependent (response) variables** and their **independent (predictor) variables**.

2. **Interdependence techniques**
   - Make **no distinction** between dependent and independent variables.
   - Treat all variables equally in a search for **underlying relationships or patterns**.

Key methods:

- **Multiple linear regression**: Predicts a single numerical dependent variable from multiple numerical independent variables.
- **Logistic regression**: Used when the outcome variable is **binary (dichotomous)**.
- **Log-linear analysis**: Models **count data** and analyzes contingency tables with **more than two variables**.
- **Analysis of covariance (ANCOVA)**: Extension of ANOVA that adds a **covariate** to see whether group differences persist after controlling for the covariate.
- **Multivariate analysis of variance (MANOVA)**: Multivariate extension of ANOVA used when there are **multiple numerical dependent variables**.
- **Exploratory factor analysis (EFA)** and **principal components analysis (PCA)**: Related techniques that reduce many metric variables into a smaller number of composite factors or components.
- **Cluster analysis**: Identifies relatively homogeneous groups (clusters) in a large number of cases **without prior information** about group membership.

Historically, the **calculation-intensive** nature of multivariate methods limited their routine use, but modern statistical software and computing power are making these techniques increasingly accessible.

---

## Introduction

- **Multivariate analysis** studies 3+ variables together for each subject.
- The real world is inherently multivariate; outcomes result from multiple inputs/influences.
- Historically underused because of computational complexity.
- With increasing computing power and user‑friendly statistical software, multivariate methods are more widely used.
- Simply knowing the names/definitions of techniques is not enough; real understanding requires:
  - Knowing what the procedure does
  - Its limitations and assumptions
  - How to interpret outputs
  - What the results actually mean
- **Mastery** comes from repeatedly applying these methods to **real datasets**.

---

## Classification of Multivariate Methods

Two main categories:

1. **Dependence techniques**
   - There is at least one **dependent/response variable** whose value is influenced by **independent/explanatory/predictor variables**.
   - Examples:
     - Multiple linear regression
     - Logistic regression
     - Discriminant function analysis
     - ANCOVA
     - MANOVA
     - Log-linear analysis
     - Probit analysis

2. **Interdependence techniques**
   - Variables are related to each other **without a clear dependent vs. independent split**.
   - All variables are treated **equally** to uncover underlying patterns.
   - Examples:
     - Factor analysis (exploratory and confirmatory)
     - Principal components analysis (PCA)
     - Cluster analysis

---

## Multiple Linear Regression

**Purpose:**

- Model the relationship between **two or more metric explanatory (independent) variables** and a **single metric response (dependent) variable**.

**General model (with n predictors):**

- \( y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \varepsilon \)
  - \(\beta_0\): constant (intercept)
  - \(\beta_i\): regression coefficients (partial regression coefficients)
  - \(\varepsilon\): random error term

**Uses:**

1. Assess the **strength of influence** of each predictor on the dependent variable.
2. Understand **how much the dependent variable changes** when predictors change.
3. Build models to **predict trends and future values**.

**Model refinement:**

- Predictors with **trivial effects** can be dropped to simplify the model.
- Software typically:
  - Estimates \(\beta_i\)s via **least squares**.
  - Tests hypotheses like \( H_0: \beta_i = 0 \) using p‑values.
  - Retains predictors with **statistically significant** coefficients.
  - Provides diagnostics for **model fit and adequacy**.

**Assumptions:**

- **Linearity** between each independent variable and the dependent variable (check with scatter plots).
- **Normally distributed residuals**.
- **Homoscedasticity** (residuals have equal variance across levels of predictors).
- **No problematic multicollinearity** (independent variables not too highly correlated with one another).

**Model fit considerations:**

- Adding predictors always increases **explained variance** (\(R^2\)).
- Too many predictors without theory can lead to **overfitting**.
- Use **adjusted \(R^2\)** and theory to guide model choice.

---

## Logistic Regression

**Setting:**

- Models a **binary (dichotomous) dependent variable** from multiple predictors.
- Predictors can be **numerical, nominal, or ordinal**.

**Applications / Questions addressed:**

- Relative importance of each predictor (strength of association).
- Can the outcome be correctly predicted from a set of predictors?
- Will the prediction generalize to new cases?
- Does adding or removing a predictor improve the model?
- Are there **interactions** between predictors?
- How good is the model at **classifying cases**?
- How well does the model **fit the data** (goodness-of-fit)?

**Variants:**

- **Binary logistic regression**: two outcome categories.
- **Multinomial (polychotomous) logistic regression**: more than two unordered categories.
- **Ordinal logistic regression**: ordered outcome categories.

**Alternatives:**

- If all predictors are continuous, normally distributed, and homoscedastic → **discriminant analysis** may be used.
- If all predictors are categorical → **logit analysis** may be used.

**Core idea – logit and odds:**

- Probabilities are nonlinear (e.g., 0.10→0.20 vs 0.80→0.90).
- Use **odds**: \( \text{odds} = p / (1 - p) \).
- **Logit (log-odds)**: \( \log_e \left( \frac{p}{1-p} \right) \).
- Logistic regression assumes a **linear relationship in the logit** (not in raw probability).

**Model form (with n predictors):**

- \( \log_e \left[ \frac{p}{1-p} \right] = \beta_0 + \beta_1 X_1 + \dots + \beta_n X_n \)
  - \(p\): probability of the event.
  - \(\beta_i\): regression coefficients.

**Interpretation of coefficients:**

- Exponentiated coefficients (\(e^{\beta_i}\)) are **odds ratios (ORs)**.
- Called **adjusted odds ratios** because they account for other predictors in the model.
- OR > 1: predictor increases odds of outcome.
- OR < 1: predictor decreases odds of outcome.
- For categorical predictors: ORs are interpreted relative to a **reference category**.
- For continuous predictors: OR corresponds to a **one-unit increase** in the predictor.

**Statistical tests and fit measures:**

- **Wald statistic** (approximate Chi-square) to test \( H_0: OR = 1 \).
- **Confidence intervals** (e.g., 95%) for OR.
- Pseudo-\(R^2\) measures:
  - Cox & Snell's \(R^2\)
  - Nagelkerke's \(R^2\)
- \(-2\) log-likelihood (\(-2LL\)): smaller values indicate better fit.
- **Goodness-of-fit tests**: e.g., Hosmer–Lemeshow test, likelihood ratio tests.
- **Classification table**: proportion of correctly classified cases.

**Strategies for building models:**

1. **Direct (enter) method**
   - All predictors entered simultaneously.
   - Used when there’s no specific hypothesis about the ordering or importance of predictors.

2. **Sequential (hierarchical) method**
   - Investigator decides the order of entry of predictors.
   - Used to evaluate the **incremental value** of new predictors given those already in the model.

3. **Stepwise method**
   - Predictors entered/removed automatically based on statistical criteria.
   - More exploratory; can be influenced by multicollinearity.
   - Often helpful to:
     - Conduct initial **univariate analyses** and **correlations** among predictors.
     - Drop weak predictors before stepwise modeling.

**Limitations and cautions:**

- Dependent variable must be **binary** (in basic logistic regression).
- **Large sample** technique:
  - Rule of thumb: at least **10 events per predictor**.
- Too many predictors with limited data → unstable models.
- **Outliers** can distort results:
  - Examine distributions and convert predictors to **z-scores**;
  - Values with |z| ≥ 3.29 often treated as outliers.
- **Statistical significance ≠ causality or clinical importance**:
  - Large samples may make trivial effects statistically significant.

---

## Discriminant Function Analysis

**Purpose:**

- Generate rules for **classifying cases** into **predefined groups** based on observed variables.
- Determine whether a set of variables **discriminates between groups**.

**Key points:**

- Commonly used for **two-group** problems using **Fisher’s linear discriminant function**.
- Produces a **classification rule** that can assign new cases to one of the groups.
- Requires a **training set** to estimate the discriminant function.
- A **classification matrix** compares observed vs. predicted group membership.
- **Performance** is assessed by the **error rate** (misclassification rate).

**Assumptions:**

- More restrictive than logistic regression (e.g., normality, equal covariance matrices across groups).
- If assumptions are not met, **logistic regression** is preferred.

---

## Other Dependence Techniques

### Analysis of Covariance (ANCOVA)

- An extension of ANOVA that includes an additional **metric independent variable (covariate)**.
- Goal: determine whether group differences in a metric dependent variable persist **after controlling for the covariate**.
- Example: comparing blood‑pressure‑lowering drugs while adjusting for **baseline blood pressure**.

### Multivariate Analysis of Variance (MANOVA)

- Extension of ANOVA for **multiple numerical dependent variables**.
- Combines dependent variables into **weighted linear composites** (canonical variates / roots).
- Tests whether the independent grouping variable explains significant variance in these **composite variables**.

### Log-linear Analysis

- Models **count data** in multiway contingency tables (3+ categorical variables).
- Models the **logarithm of expected cell counts** as a linear function of parameters.
- Parameters represent:
  - Pairwise associations between variables.
  - Higher-order interactions among variables.

### Probit Analysis

- Common in **toxicology**.
- Relates **proportion responding** (e.g., mortality/survival) to **dose**.
- Uses a **probit transformation** of proportions and models it as a **linear function of dose** (often log-dose).
- Parameters are estimated by **maximum likelihood**.
- Closely related to the **logit model** used in logistic regression.

---

## Factor Analysis and Principal Components Analysis (PCA)

### Factor Analysis

**Purpose:**

- Reduce a large number of metric variables to a smaller number of **composite factors**.
- Identify underlying **latent variables** that explain correlations among observed (manifest) variables.
- Widely used in **psychometrics, social sciences, and market research**.

**Types:**

1. **Exploratory Factor Analysis (EFA)**
   - Used when there is **no prior knowledge** of the number or nature of the latent dimensions.
   - Example: assessing **patient satisfaction** with multiple observed items.
   - Explores the **correlation matrix** to see how variables cluster.

2. **Confirmatory Factor Analysis (CFA)**
   - Used when there is a **prior hypothesis** about how items load on latent factors.
   - Example: verifying whether items group into **physical fatigue** and **mental fatigue**.
   - Often applied to **new data** to confirm constructs found by EFA.
   - Mathematically more demanding; closely related to **structural equation modeling (SEM)**.

**Technical notes:**

- **Common factors** are latent variables explaining correlation among observed variables.
- **Factor loadings**: regression coefficients of observed variables on the common factors.
- **Factor rotation** is used to achieve **simple structure** (each factor influences only a subset of variables), improving interpretability.

### Principal Components Analysis (PCA)

**Purpose:**

- Transform original variables into new, **uncorrelated principal components**.
- Each component is a **linear combination** of the original variables.

**Key properties:**

- Components are ordered by the **amount of variance they explain**:
  - 1st component: explains the **largest variance**.
  - 2nd component: explains the next largest portion **not accounted for** by the 1st.
  - And so on.
- In practice, only a **small number of components** are retained.

**Eigenvalues and Scree Plot:**

- **Eigenvalues** measure the amount of variance explained by each component.
- **Scree plot** (eigenvalue vs. component number) helps decide how many components to retain.

**Relationship between EFA and PCA:**

- Often implemented side‑by‑side in software, but **not identical**.
- PCA is earlier and conceptually simpler (Karl Pearson, 1901).
- Recommended guidance:
  - Use **factor analysis** when you have **theoretical expectations** about latent constructs.
  - Use **PCA** when the goal is **data reduction** and pattern exploration rather than modeling latent constructs.

---

## Cluster Analysis

**Purpose:**

- Classify an initially **unlabeled set of cases** into relatively homogeneous **clusters**.
- No prior information about group membership.
- Also called **classification analysis** or **numerical taxonomy**.

**Steps in cluster analysis:**

1. Formulate the problem and **select variables** (based on hypotheses and prior knowledge).
2. Choose a **distance or similarity measure** (often Euclidean distance or squared Euclidean distance).
3. Choose a **clustering procedure**:
   - **Hierarchical** (agglomerative or divisive)
   - **Nonhierarchical** (K-means)
   - **Two-step** methods
4. Decide on the **number of clusters**.
5. Interpret **cluster profiles**.
6. Assess **validity** of clusters.

**Hierarchical clustering:**

- Does **not** require pre‑specifying the number of clusters.
- Results shown using a **dendrogram** (tree diagram).
  - Nodes: objects
  - Branches: how objects group into clusters
  - Branch length: distance between merged clusters

**K-means (nonhierarchical) clustering:**

- Requires pre‑specifying the number of clusters (K).

**Cautions:**

- Software will **always** generate clusters, even if **no real structure** exists.
- Many choices (distance measure, linkage method, number of clusters) can bias results.
- Risk of **data dredging** to find preferred structure.
- Interpret results **with great caution**.

---

## Sample Size in Multivariate Analysis

- Most multivariate techniques are essentially **large-sample methods**.
- Applying them to **small datasets** can yield unstable or unreliable results.
- There are **no simple universal formulas** for determining sample size in most multivariate methods.

**For regression analyses (especially logistic regression):**

- Common rule: **≥ 10 events of the outcome per predictor variable** in the model.
- This rule may be inadequate when many **categorical covariates** are present.
- Even when sample size appears sufficient, perform **model quality checks** before using results for clinical or practical decisions.

---

## Overall Takeaways

- Multivariate methods allow researchers to model **complex, realistic relationships** involving multiple variables.
- Advances in statistical software and computing are making these methods increasingly accessible.
- Responsible use requires:
  - Understanding **assumptions and limitations**
  - Checking **model fit and diagnostics**
  - Being cautious about **overfitting** and **multicollinearity**
  - Distinguishing **statistical significance** from **clinical or causal significance**
