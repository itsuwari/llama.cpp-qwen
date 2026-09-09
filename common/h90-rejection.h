#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>
struct h90_prob { int32_t id; double p; };
using h90_distribution = std::vector<h90_prob>;
struct h90_draft_step { int32_t sampled; h90_distribution q; };
inline h90_distribution h90_normalize(h90_distribution v) {
    std::sort(v.begin(),v.end(),[](const h90_prob&a,const h90_prob&b){return a.id<b.id;});
    double sum=0;
    for(size_t i=0;i<v.size();++i) {
        if(v[i].id<0 || !std::isfinite(v[i].p) || v[i].p<0 || (i && v[i-1].id==v[i].id))
            throw std::runtime_error("Invalid sparse categorical distribution");
        sum+=v[i].p;
    }
    if(!(sum>0) || !std::isfinite(sum))throw std::runtime_error("Empty categorical support");
    for(auto &x:v)x.p/=sum;
    return v;
}
inline double h90_prob_at(const h90_distribution &v,int32_t id) {
    auto it=std::lower_bound(v.begin(),v.end(),id,[](const h90_prob&a,int32_t b){return a.id<b;});
    return it!=v.end() && it->id==id ? it->p : 0;
}
inline int32_t h90_categorical(const h90_distribution&v,double u) {
    if(!(u>=0 && u<1))throw std::runtime_error("Uniform variate outside [0,1)");
    double sum=0;int32_t last=-1;
    for(const auto &x:v)if(x.p>0){sum+=x.p;last=x.id;if(u<sum)return x.id;}
    if(last<0)throw std::runtime_error("No categorical outcome");
    return last;
}
struct h90_rejection_result {int32_t token;bool accepted;};
inline h90_rejection_result h90_reject_step(const h90_distribution &p,const h90_distribution &q,
                                            int32_t y,double accept_u,double residual_u) {
    const double qy=h90_prob_at(q,y),py=h90_prob_at(p,y);
    if(!(qy>0) || !(accept_u>=0 && accept_u<1))throw std::runtime_error("Invalid rejection proposal");
    if(accept_u<std::min(1.0,py/qy))return {y,true};
    h90_distribution residual;residual.reserve(p.size());
    for(const auto &x:p){const double r=std::max(0.0,x.p-h90_prob_at(q,x.id));if(r>0)residual.push_back({x.id,r});}
    return {h90_categorical(h90_normalize(std::move(residual)),residual_u),false};
}
