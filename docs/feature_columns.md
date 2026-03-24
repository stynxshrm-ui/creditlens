Here's the complete table:

| Column | Meaning | Logic (Why Important) | Allowed? |
|--------|---------|----------------------|---------|
| **id** | Unique loan identifier | Just for tracking | ✅ Allowed (harmless) |
| **issue_d** | Loan funding date | Helps with time trends | ⚠️ Allowed only for splitting data, not as feature |
| **funded_amnt** | Amount actually given | Bigger loans = more risk | ✅ Allowed |
| **loan_amnt** | Amount requested | Same as funded_amnt usually | ✅ Allowed |
| **term** | Loan length (36/60 months) | Longer = more time to default | ✅ Allowed |
| **int_rate** | Interest rate (%) | Higher = bank flagged as risky | ✅ Allowed |
| **installment** | Monthly payment amount | Too high relative to income = struggle | ✅ Allowed |
| **grade** | Risk rating (A-G) | A = safe, G = risky | ✅ Allowed |
| **sub_grade** | Detailed risk rating (A1-G5) | Finer risk distinction | ✅ Allowed |
| **purpose** | Why borrower wants loan | Some purposes (medical, debt consolidation) = higher risk | ✅ Allowed |
| **title** | Borrower's description | Same as purpose, free text | ✅ Allowed (with NLP) |
| **initial_list_status** | How investors funded it | Technical detail, low importance | ✅ Allowed |
| **application_type** | Individual or Joint | Joint = two incomes = safer | ✅ Allowed |
| **verification_status** | Income verified or not | Unverified = might be lying | ✅ Allowed |
| **dti** | Debt-to-income ratio | Higher = overextended = risk | ✅ Allowed |
| **delinq_2yrs** | Late payments in past 2 years | Recent misses = high risk | ✅ Allowed |
| **earliest_cr_line** | Age of credit history | Older = experienced = safer | ✅ Allowed |
| **open_acc** | Number of open credit accounts | Too many = overextended | ✅ Allowed |
| **pub_rec** | Public records (liens, judgments) | Any = financial trouble | ✅ Allowed |
| **revol_bal** | Credit card balances | High = relying on credit | ✅ Allowed |
| **revol_util** | Credit card utilization % | >70% = desperate = risk | ✅ Allowed |
| **total_acc** | Total accounts ever opened | Shows credit experience | ✅ Allowed |
| **mort_acc** | Number of mortgages | Homeowners = more stable | ✅ Allowed |
| **pub_rec_bankruptcies** | Bankruptcies filed | Extremely high risk | ✅ Allowed |
| **out_prncp** | Principal still owed | Shows current status | ❌ NOT Allowed (leakage) |
| **out_prncp_inv** | Principal owed to investors | Same as above | ❌ NOT Allowed (leakage) |
| **total_pymnt** | Total payments received | Reveals if loan was paid | ❌ NOT Allowed (leakage) |
| **total_rec_prncp** | Principal repaid | Directly shows outcome | ❌ NOT Allowed (leakage) |
| **total_rec_int** | Interest repaid | Low = stopped early | ❌ NOT Allowed (leakage) |
| **total_rec_late_fee** | Late fees collected | High = missed payments | ❌ NOT Allowed (leakage) |
| **recoveries** | Money collected after default | Only known after default | ❌ NOT Allowed (leakage) |
| **collection_recovery_fee** | Fees to collection agencies | Only known after default | ❌ NOT Allowed (leakage) |
| **loan_status** | Final outcome (Fully Paid, Charged Off, etc.) | This is your **target variable** | ✅ Allowed (as target only) |

## Summary
- **Allowed**: 23 columns for prediction
- **Not Allowed**: 9 columns that cause leakage
- **Target**: `loan_status` is what we're predicting
