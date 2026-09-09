#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cstdint>

int main(int argc,char **argv) {
 try {
  if(argc!=4) throw std::runtime_error("usage: replay model tokens.txt output.f32");
  std::ifstream f(argv[2]); std::vector<llama_token> tokens; int token;
  while(f>>token) tokens.push_back(token);
  if(tokens.size()<4608) throw std::runtime_error("fixture requires 4608 tokens");
  ggml_backend_load_all(); llama_backend_init();
  auto mp=llama_model_default_params();mp.n_gpu_layers=999;
  std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
  if(!model) throw std::runtime_error("model load failed");
  auto cp=llama_context_default_params();cp.n_ctx=8192;cp.n_batch=512;cp.n_ubatch=512;
  cp.n_seq_max=1;cp.n_rs_seq=8;cp.n_threads=24;cp.n_threads_batch=24;
  cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q8_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
  std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
  if(!ctx) throw std::runtime_error("context load failed");
  llama_set_embeddings_nextn(ctx.get(),true,false);
  const int nv=llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
  auto mem=llama_get_memory(ctx.get());
  std::vector<llama_pos> positions(512);
  std::vector<int8_t> logits(512);
  auto decode=[&](int pos,int n,bool outputs){
   for(int i=0;i<n;++i){positions[i]=pos+i;logits[i]=outputs?1:0;}
   llama_batch b{};b.n_tokens=n;b.token=tokens.data()+pos;b.pos=positions.data();b.logits=logits.data();
   if(llama_decode(ctx.get(),b)!=0)throw std::runtime_error("decode failed");
   llama_synchronize(ctx.get());
  };
  auto prefill=[&](){for(int pos=0;pos<4096;pos+=512)decode(pos,512,false);};
  prefill();int pos=4096,save_pos=-1;
  std::vector<uint8_t> saved;
  const int widths[]={6,8,4,2,7,3,5};
  const int accepts[]={1,3,4,1,6,2,5,6};
  std::ofstream out(argv[3],std::ios::binary);out.write(reinterpret_cast<const char *>(&nv),sizeof(nv));
  for(int round=0;round<32;++round){
   if(round==8){
    const size_t n=llama_state_seq_get_size(ctx.get(),0);
    if(n>size_t(2)*1024*1024*1024)throw std::runtime_error("state unexpectedly large");
    saved.resize(n);if(llama_state_seq_get_data(ctx.get(),saved.data(),n,0)!=n)throw std::runtime_error("state save failed");
    save_pos=pos;
   }
   if(round==16){
    if(llama_state_seq_set_data(ctx.get(),saved.data(),saved.size(),0)==0)throw std::runtime_error("state restore failed");
    pos=save_pos;
   }
   if(round==24){llama_memory_clear(mem,true);prefill();pos=4096;}
   const int n=widths[round%7],keep=std::min(n,accepts[round%8]);decode(pos,n,true);
   const float * values=llama_get_logits_ith(ctx.get(),n-1);if(!values)throw std::runtime_error("missing logits");
   int32_t meta[]={round,pos,n,keep};out.write(reinterpret_cast<const char *>(meta),sizeof(meta));
   out.write(reinterpret_cast<const char *>(values),size_t(nv)*sizeof(float));
   if(!llama_memory_seq_rm(mem,0,pos+keep,-1))throw std::runtime_error("suffix rollback failed");
   pos+=keep;
   fprintf(stderr,"REPLAY round=%d pos=%d width=%d keep=%d\n",round,pos,n,keep);
  }
  if(!out)throw std::runtime_error("write failed");
  fprintf(stderr,"REPLAY_PASS rounds=32 restores=1 clears=1 vocab=%d\n",nv);return 0;
 }catch(const std::exception &e){fprintf(stderr,"REPLAY_ERROR %s\n",e.what());return 1;}
}
