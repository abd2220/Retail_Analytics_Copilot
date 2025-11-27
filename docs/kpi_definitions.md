# KPI Definitions
## Average Order Value (AOV)
- AOV = SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID)

## Gross Margin
- GM = SUM((UnitPrice - CostOfGoods) * Quantity * (1 - Discount))
- Since CostOfGoods is missing in Northwind, use this approximation: CostOfGoods = 0.7 * UnitPrice