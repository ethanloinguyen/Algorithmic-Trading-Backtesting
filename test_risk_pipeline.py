import sys, pathlib
sys.path.insert(0, str(pathlib.Path("backend")))

from app.services.pipeline_service import run_risk_pipeline

result = run_risk_pipeline(
    user_portfolio=["AAPL", "MSFT", "GOOGL"],
    horizon_days=63,
    n_sims=500,
)

print("\n=== RECOMMENDATIONS ===")
for r in result["recommendations"]:
    print(f"  [{r['sector']:<30}]  {r['stock']}  dcor={r['avg_dcor_to_portfolio']:.4f}")

print("\n=== PORTFOLIO RISK (63-day horizon) ===")
p = result["risk"]["portfolio"]
print(f"  VaR 95%:           {p['var_95']:.2%}")
print(f"  CVaR 95%:          {p['cvar_95']:.2%}")
print(f"  Expected drawdown: {p['expected_max_drawdown']:.2%}")
print(f"  Prob of loss:      {p['prob_loss']:.2%}")
print(f"  Diversif. benefit: {p['diversification_benefit_95']:.2%}")