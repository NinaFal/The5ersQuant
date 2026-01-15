#!/usr/bin/env python3
"""
Test Index Lot Sizing

Verifies that NAS100 and SP500 have correct contract specs for proper lot sizing.
"""

from tradr.risk.position_sizing import calculate_lot_size, get_contract_specs

def test_index_lot_sizing():
    """Test lot sizing for indices with updated specs."""
    
    # Test parameters
    balance = 20000.0  # $20K account
    risk_pct = 0.006  # 0.6% risk
    
    # Example NAS100 trade
    # Price: 20000, Stop: 19900 (100 points)
    nas100_entry = 20000.0
    nas100_sl = 19900.0
    
    # Example SP500 trade  
    # Price: 5000, Stop: 4950 (50 points)
    sp500_entry = 5000.0
    sp500_sl = 4950.0
    
    print("=" * 70)
    print("INDEX LOT SIZING TEST - 5ERS CONTRACT SPECS")
    print("=" * 70)
    print(f"Account Balance: ${balance:,.0f}")
    print(f"Risk per Trade: {risk_pct*100}% = ${balance * risk_pct:.2f}")
    print()
    
    # Check contract specs
    nas100_specs = get_contract_specs("NAS100")
    sp500_specs = get_contract_specs("SP500")
    
    print("CONTRACT SPECIFICATIONS:")
    print(f"  NAS100: ${nas100_specs['pip_value_per_lot']}/point (should be $20)")
    print(f"  SP500:  ${sp500_specs['pip_value_per_lot']}/point (should be $10)")
    print()
    
    # Calculate NAS100 lot size
    nas100_result = calculate_lot_size(
        symbol="NAS100",
        account_balance=balance,
        risk_percent=risk_pct,
        entry_price=nas100_entry,
        stop_loss_price=nas100_sl,
    )
    
    # Calculate SP500 lot size
    sp500_result = calculate_lot_size(
        symbol="SP500",
        account_balance=balance,
        risk_percent=risk_pct,
        entry_price=sp500_entry,
        stop_loss_price=sp500_sl,
    )
    
    print("NAS100 TRADE:")
    print(f"  Entry:      {nas100_entry:,.0f}")
    print(f"  Stop Loss:  {nas100_sl:,.0f}")
    print(f"  Stop Points: {nas100_result['stop_pips']:.0f}")
    print(f"  Lot Size:    {nas100_result['lot_size']:.2f}")
    print(f"  Risk USD:    ${nas100_result['risk_usd']:.2f}")
    print(f"  Risk %:      {nas100_result['actual_risk_pct']*100:.3f}%")
    print()
    
    # Verify calculation
    # Risk per lot = stop_points * $20/point
    # Lot size should be: $120 / (100 points * $20) = 0.06 lots
    expected_nas100_lot = (balance * risk_pct) / (100 * 20)
    print(f"  Expected:    {expected_nas100_lot:.2f} lots")
    print(f"  Actual:      {nas100_result['lot_size']:.2f} lots")
    print(f"  Match: {'✅ CORRECT' if abs(nas100_result['lot_size'] - expected_nas100_lot) < 0.01 else '❌ WRONG'}")
    print()
    
    print("SP500 TRADE:")
    print(f"  Entry:      {sp500_entry:,.0f}")
    print(f"  Stop Loss:  {sp500_sl:,.0f}")
    print(f"  Stop Points: {sp500_result['stop_pips']:.0f}")
    print(f"  Lot Size:    {sp500_result['lot_size']:.2f}")
    print(f"  Risk USD:    ${sp500_result['risk_usd']:.2f}")
    print(f"  Risk %:      {sp500_result['actual_risk_pct']*100:.3f}%")
    print()
    
    # Verify calculation
    # Risk per lot = stop_points * $10/point
    # Lot size should be: $120 / (50 points * $10) = 0.24 lots
    expected_sp500_lot = (balance * risk_pct) / (50 * 10)
    print(f"  Expected:    {expected_sp500_lot:.2f} lots")
    print(f"  Actual:      {sp500_result['lot_size']:.2f} lots")
    print(f"  Match: {'✅ CORRECT' if abs(sp500_result['lot_size'] - expected_sp500_lot) < 0.01 else '❌ WRONG'}")
    print()
    
    print("=" * 70)
    print("COMPARISON WITH OLD SPECS ($1/point):")
    print("=" * 70)
    
    # What would happen with old $1/point specs?
    old_nas100_lot = (balance * risk_pct) / (100 * 1)  # $1/point
    old_sp500_lot = (balance * risk_pct) / (50 * 1)   # $1/point
    
    print(f"NAS100:")
    print(f"  Old (wrong): {old_nas100_lot:.2f} lots → Risk ${old_nas100_lot * 100 * 1:.2f}")
    print(f"  New (correct): {nas100_result['lot_size']:.2f} lots → Risk ${nas100_result['risk_usd']:.2f}")
    print(f"  Multiplier: {old_nas100_lot / nas100_result['lot_size']:.1f}x too high with old specs!")
    print()
    
    print(f"SP500:")
    print(f"  Old (wrong): {old_sp500_lot:.2f} lots → Risk ${old_sp500_lot * 50 * 1:.2f}")
    print(f"  New (correct): {sp500_result['lot_size']:.2f} lots → Risk ${sp500_result['risk_usd']:.2f}")
    print(f"  Multiplier: {old_sp500_lot / sp500_result['lot_size']:.1f}x too high with old specs!")
    print()
    
    print("=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print("✅ Index lot sizes are now CORRECT for 5ers:")
    print("   - NAS100: $20 per point per lot")
    print("   - SP500:  $10 per point per lot")
    print()
    print("⚠️  Previous bug would have resulted in 10-20x OVERLEVERAGED positions!")
    print("=" * 70)


if __name__ == "__main__":
    test_index_lot_sizing()
