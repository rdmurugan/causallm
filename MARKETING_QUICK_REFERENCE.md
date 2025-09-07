# Marketing Attribution - Quick Reference

## 🚀 Getting Started

```python
from causallm.domains.marketing import MarketingDomain

# Initialize (with performance optimizations)
marketing = MarketingDomain(enable_performance_optimizations=True)

# Generate sample data
data = marketing.generate_marketing_data(
    n_customers=10000,
    n_touchpoints=30000
)

# Run attribution analysis
result = marketing.analyze_attribution(
    data, 
    model='data_driven'  # Recommended for most cases
)

print(f"Top channel: {max(result.channel_attribution.items(), key=lambda x: x[1])}")
```

## 📊 Attribution Models

| Model | Best For | Description |
|-------|----------|-------------|
| `first_touch` | Brand awareness | 100% credit to first interaction |
| `last_touch` | Direct response | 100% credit to last interaction |
| `linear` | Balanced view | Equal credit across touchpoints |
| `time_decay` | Recency matters | More credit to recent interactions |
| `position_based` | Funnel analysis | Credit to first/last + middle |
| `data_driven` | **Recommended** | Uses causal inference |
| `shapley` | Advanced | Game theory based |

## 🎯 Common Use Cases

### Multi-Touch Attribution
```python
# Compare multiple models
models = ['first_touch', 'last_touch', 'data_driven']
results = {}
for model in models:
    results[model] = marketing.analyze_attribution(data, model=model)
```

### Campaign ROI Analysis
```python
spend_data = marketing.data_generator.generate_campaign_spend_data()
roi_results = marketing.attribution_analyzer.analyze_campaign_roi(
    data=data, 
    spend_data=spend_data
)
```

### Industry-Specific Analysis
```python
from causallm.domains.marketing.templates.attribution_template import AttributionAnalysisTemplate

template = AttributionAnalysisTemplate()

# E-commerce
ecommerce_result = template.run_analysis('ecommerce_funnel', data)

# B2B
b2b_result = template.run_analysis('lead_generation', data)

# Brand
brand_result = template.run_analysis('brand_awareness', data)
```

## ⚡ Performance Features

```python
# Large datasets (auto-optimized)
marketing = MarketingDomain(enable_performance_optimizations=True)

# Caching for repeated analysis
from causallm import EnhancedCausalLLM
causal_llm = EnhancedCausalLLM(cache_dir="./cache")

# Async model comparison
results = marketing.attribution_analyzer.compare_attribution_models(
    data, models=['first_touch', 'last_touch', 'linear']
)
```

## 📈 Business Insights

```python
# Get domain knowledge
knowledge = marketing.knowledge

# Industry benchmarks
benchmarks = knowledge.get_industry_benchmarks('ecommerce')

# Attribution recommendations
recs = knowledge.get_model_recommendations({
    'type': 'ecommerce',
    'sales_cycle_days': 14
})

# Channel insights
channel_info = knowledge.get_channel_characteristics('paid_search')
```

## 🔧 Advanced Features

### Cross-Device Attribution
```python
cross_device_result = template.run_analysis('cross_device', data)
```

### Seasonal Analysis
```python
seasonal_result = template.run_analysis('seasonal_attribution', data)
```

### Campaign Templates
```python
from causallm.domains.marketing.templates.campaign_analysis import CampaignAnalysisTemplate

campaign_template = CampaignAnalysisTemplate()
effectiveness = campaign_template.run_analysis('campaign_effectiveness', data)
```

## 📋 Quick Tips

1. **Start with `data_driven` model** - best for most scenarios
2. **Use 30-day attribution window** for e-commerce
3. **Enable performance optimizations** for >10K touchpoints
4. **Cache results** for repeated analysis
5. **Compare multiple models** for validation

## 📚 Examples

- **Quick Start**: `examples/marketing_attribution_quickstart.py`
- **Comprehensive**: `examples/comprehensive_marketing_attribution_demo.py`
- **Guide**: `examples/MARKETING_ATTRIBUTION_GUIDE.md`

## 🆘 Common Issues

| Issue | Solution |
|-------|----------|
| Import error | `pip install causallm[full]` |
| Performance slow | Enable optimizations: `MarketingDomain(enable_performance_optimizations=True)` |
| Memory error | Use data chunking or reduce dataset size |
| Low accuracy | Check attribution window and data quality |

---

💡 **Pro Tip**: Always validate attribution results with business intuition and run incrementality tests when possible.