def sip(investment, tenure, interest, amount=0, is_year=True, is_percent=True, show_amount_list=False):
    tenure = tenure*12 if is_year else tenure
    interest = interest/100 if is_percent else interest
    interest /= 12
    amount_every_month = {}
    for month in range(tenure):
        amount = (amount + investment)*(1+interest)
        amount_every_month[month+1] = f'{amount:.2f}'
    return {'Amount @ Maturity': round(amount), 'Amount every month': amount_every_month} if show_amount_list else {'Amount @ Maturity': amount} 

def emi(amount, tenure, interest, is_year=True, is_percent=True):
    tenure = tenure*12 if is_year else tenure
    interest = interest/100 if is_percent else interest
    interest /= 12
    emi = (amount*interest*(1+interest)**tenure) / ((1+interest)**tenure-1)
    total_payable = emi*tenure
    interest_amount = total_payable - amount
    return {'EMI': f'{emi:.2f}', 'Total Repayment Amount': f'{total_payable:.2f}', 'Interest Amount': f'{interest_amount:.2f}'}