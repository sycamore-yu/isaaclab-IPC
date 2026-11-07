// Exact comparisons against the public CPU graph; no timing measurements.
#include <distance_map_planner/distance_map_planner.h>
#ifdef PATHCOVER_TEST_CUDA
#include "pathcover_distance_map.h"
#include <cuda_runtime.h>
#endif
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Enumerate every lattice point in the closed axis-aligned box swept by one
// grid edge. This does not use the implementation's precomputed subset masks.
std::vector<uint32_t> oracle(const std::vector<int8_t>& potential, const Vec3i& dim) {
    std::vector<uint32_t> masks(potential.size());
    auto id = [&](int x, int y, int z) { return x + dim.x() * (y + dim.y() * z); };
    for (int z = 0; z < dim.z(); ++z) for (int y = 0; y < dim.y(); ++y) for (int x = 0; x < dim.x(); ++x) {
        if (potential[id(x,y,z)] >= 100) continue;
        for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
            if (!dx && !dy && !dz) continue;
            bool free = true;
            for (int sx = std::min(0,dx); sx <= std::max(0,dx); ++sx)
                for (int sy = std::min(0,dy); sy <= std::max(0,dy); ++sy)
                    for (int sz = std::min(0,dz); sz <= std::max(0,dz); ++sz) {
                        if (!sx && !sy && !sz) continue;
                        const int xx=x+sx, yy=y+sy, zz=z+sz;
                        if (xx<0 || xx>=dim.x() || yy<0 || yy>=dim.y() || zz<0 || zz>=dim.z()
                                || potential[id(xx,yy,zz)]>=100) free=false;
                    }
            if (free) masks[id(x,y,z)] |= uint32_t{1} << ((dx+1)*9+(dy+1)*3+dz+1);
        }
    }
    return masks;
}

void compareStates(const std::vector<DMP::StatePtr>& a, const std::vector<DMP::StatePtr>& b) {
    require(a.size()==b.size(), "state count differs");
    for (std::size_t i=0; i<a.size(); ++i) {
        require(a[i]->id==b[i]->id && a[i]->parentId==b[i]->parentId, "state/path order or parent differs");
        require(a[i]->x==b[i]->x && a[i]->y==b[i]->y && a[i]->z==b[i]->z, "state coordinates differ");
        require(std::memcmp(&a[i]->g,&b[i]->g,sizeof(double))==0 && std::memcmp(&a[i]->h,&b[i]->h,sizeof(double))==0,
                "state g/h bits differ");
        require(a[i]->opened==b[i]->opened && a[i]->closed==b[i]->closed, "state flags differ");
    }
}
void compareGraphs(DMP::GraphSearch& a, double ca, DMP::GraphSearch& b, double cb) {
    require(std::memcmp(&ca,&cb,sizeof(double))==0, "path cost bits differ");
    require(a.getStartAnchorIndex()==b.getStartAnchorIndex() && a.getGoalAnchorIndex()==b.getGoalAnchorIndex(), "anchor selection differs");
    compareStates(a.getPath(), b.getPath());
    compareStates(a.getAllSet(), b.getAllSet());
}

void compareThreeSearches(const std::vector<int8_t>& potential, const Vec3i& dim,
                         const std::vector<uint32_t>& masks) {
    DMP::GraphSearch cpu(potential.data(),dim.x(),dim.y(),dim.z(),1.5,.1);
    DMP::GraphSearch cached(potential.data(),dim.x(),dim.y(),dim.z(),1.5,.1,false,masks.data());
    const int nx=dim.x(), ny=dim.y(), nz=dim.z();
    double a=cpu.plan(0,0,0,nx-1,ny-1,nz-1), b=cached.plan(0,0,0,nx-1,ny-1,nz-1);
    compareGraphs(cpu,a,cached,b);
    const std::vector<DMP::GridAnchor> starts{{0,0,0,.125},{1,1,0,2.25},{nx-1,0,0,7.75}};
    const std::vector<DMP::GridAnchor> goals{{nx-1,ny-1,nz-1,1.125},{nx-2,ny-1,nz-1,.5}};
    a=cpu.plan(starts,goals); b=cached.plan(starts,goals); compareGraphs(cpu,a,cached,b);
    std::vector<bool> region(potential.size(),true);
    for (int z=0; z<nz; ++z) for (int y=4; y<7; ++y) for (int x=5; x<8; ++x)
        region[x+nx*(y+ny*z)]=false;
    a=cpu.plan(0,0,0,nx-1,ny-1,nz-1,region); b=cached.plan(0,0,0,nx-1,ny-1,nz-1,region);
    compareGraphs(cpu,a,cached,b);
}

std::vector<int8_t> fixtureMap(int fixture, const Vec3i& dim, bool occupancy) {
    std::vector<int8_t> cells(dim.prod());
    auto id=[&](int x,int y,int z){return x+dim.x()*(y+dim.y()*z);};
    for (int z=0; z<dim.z(); ++z) for (int y=0; y<dim.y(); ++y) for (int x=0; x<dim.x(); ++x) {
        const unsigned hash=unsigned(x*19+y*31+z*43+fixture*67);
        cells[id(x,y,z)]=hash%29<3?100:(occupancy?0:hash%11);
    }
    for (const auto& p:std::vector<Vec3i>{{0,0,0},{1,1,0},{dim.x()-1,0,0},
            {dim.x()-1,dim.y()-1,dim.z()-1},{dim.x()-2,dim.y()-1,dim.z()-1}})
        cells[id(p.x(),p.y(),p.z())]=0;
    return cells;
}

void cpuEquivalence() {
    const Vec3i dim(13,11,5);
    for (int fixture=0; fixture<6; ++fixture) {
        const auto potential=fixtureMap(fixture,dim,false);
        compareThreeSearches(potential,dim,oracle(potential,dim));
    }
}

void lifecycle() {
    const Vec3i dim(7,5,3);
    auto map=std::make_shared<JPS::MapUtil<3>>();
    std::vector<int8_t> open(dim.prod(),0), wall=open;
    for (int z=0; z<dim.z(); ++z) for (int y=0; y<dim.y(); ++y) wall[3+dim.x()*(y+dim.y()*z)]=100;
    map->setMap(Vec3f::Zero(),dim,open,.25);
    const Vec3f start=map->intToFloat(Vec3i(1,2,1)), goal=map->intToFloat(Vec3i(5,2,1));
    DMPlanner3D planner; planner.setEps(1.5); planner.setPotentialRadius(Vec3f::Constant(.5));
    const auto mask=oracle(open,dim);
    planner.setPotentialMap(map,open,mask);
    require(planner.computeGlobalRawPath(start,goal), "initial masked path failed");
    planner.setPotentialMap(map,wall);
    require(!planner.computeGlobalRawPath(start,goal), "two-argument installation retained stale masks");
    planner.setPotentialMap(map,open,mask);
    map->setMap(Vec3f::Zero(),dim,wall,.25);
    planner.setMap(map,start);
    require(!planner.computeGlobalRawPath(start,goal), "CPU setMap retained stale masks");
    map->setMap(Vec3f::Zero(),dim,open,.25);
    planner.setPotentialMap(map,open,mask);
    require(planner.computeGlobalRawPath(start,goal), "map reinstallation retained old occupancy");
    bool rejected=false;
    try { planner.setPotentialMap(map,open,{1}); } catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"incorrect mask size was accepted");
    std::vector<int8_t> changed_cost=open;
    for(std::size_t i=0;i<changed_cost.size();++i) changed_cost[i]=i%97;
    DMP::GraphSearch cpu(changed_cost.data(),dim.x(),dim.y(),dim.z(),1.5,.1);
    DMP::GraphSearch cached(changed_cost.data(),dim.x(),dim.y(),dim.z(),1.5,.1,false,mask.data());
    const double a=cpu.plan(1,2,1,5,2,1),b=cached.plan(1,2,1,5,2,1); compareGraphs(cpu,a,cached,b);
}

#ifdef PATHCOVER_TEST_CUDA
std::vector<std::array<int,4>> dilationMask(DMPlanner3D& cpu, double resolution) {
    std::vector<std::array<int,4>> mask;
    for (const auto& entry:cpu.potentialMask(resolution))
        mask.push_back({entry.first.x(),entry.first.y(),entry.first.z(),entry.second});
    return mask;
}
pathcover_cuda::DistanceMapResult checkGpu(pathcover_cuda::DistanceMap& gpu, const std::vector<int8_t>& occupancy,
                                          const Vec3i& dim, double resolution=.25) {
    auto map=std::make_shared<JPS::MapUtil<3>>(); map->setMap(Vec3f::Zero(),dim,occupancy,resolution);
    DMPlanner3D cpu; cpu.setPotentialRadius(Vec3f::Constant(1.0)); cpu.setMap(map,Vec3f::Zero());
    auto result=gpu.compute(occupancy,{dim.x(),dim.y(),dim.z()},dilationMask(cpu,resolution));
    require(result.potential==cpu.getMapUtil()->map_, "GPU potential changed from original CPU output");
    require(result.legal_moves==oracle(result.potential,dim), "GPU legal mask differs from direct lattice-box oracle");
    return result;
}
void gpuMasks() {
    pathcover_cuda::DistanceMap gpu;
    for (const auto& dim:std::vector<Vec3i>{{1,1,1},{1,7,3},{7,1,3},{7,5,1},{9,7,5},{3,3,3}}) {
        const std::array<int8_t,7> values{{-127,-1,0,1,7,99,100}};
        std::vector<int8_t> occupancy(dim.prod());
        for(std::size_t i=0;i<occupancy.size();++i)occupancy[i]=values[i%values.size()];
        checkGpu(gpu,occupancy,dim);
        std::fill(occupancy.begin(),occupancy.end(),int8_t{0}); checkGpu(gpu,occupancy,dim);
        // Six virtual workspace faces; changing the same shape must refresh masks.
        for(int z=0;z<dim.z();++z)for(int y=0;y<dim.y();++y)for(int x=0;x<dim.x();++x)
            if(x==0||x==dim.x()-1||y==0||y==dim.y()-1||z==0||z==dim.z()-1)
                occupancy[x+dim.x()*(y+dim.y()*z)]=100;
        checkGpu(gpu,occupancy,dim);
    }
    const Vec3i dim(3,3,3); std::vector<int8_t> cage(dim.prod());
    for(const auto& d:std::vector<Vec3i>{{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}})
        cage[1+d.x()+3*(1+d.y()+3*(1+d.z()))]=100;
    auto result=checkGpu(gpu,cage,dim); require(result.legal_moves[13]==0,"six-axis cage allowed a diagonal escape");
}
void gpuEquivalence() {
    pathcover_cuda::DistanceMap gpu; const Vec3i dim(13,11,5);
    for(int fixture=0;fixture<6;++fixture) {
        auto result=checkGpu(gpu,fixtureMap(fixture,dim,true),dim);
        compareThreeSearches(result.potential,dim,result.legal_moves);
    }
}
void frozen(const char* parameters, const char* coarse_path, const char* mask_path) {
    std::ifstream settings(parameters); Vec3i dim; Vec3f origin,start,goal;
    double resolution,weight,cweight,radius; int ordinary;
    settings>>dim.x()>>dim.y()>>dim.z()>>resolution>>weight>>cweight>>radius>>ordinary;
    for(auto* p:{&origin,&start,&goal})for(int axis=0;axis<3;++axis)settings>>(*p)[axis];
    require(bool(settings) && ordinary,"frozen parameters must have ordinary endpoints");
    std::vector<int8_t> occupancy(dim.prod()); std::ifstream file(coarse_path,std::ios::binary);
    file.read(reinterpret_cast<char*>(occupancy.data()),occupancy.size());require(bool(file),"frozen map read failed");
    pathcover_cuda::DistanceMap gpu; auto field=checkGpu(gpu,occupancy,dim,resolution);
    auto map=std::make_shared<JPS::MapUtil<3>>();map->setMap(origin,dim,occupancy,resolution);
    const Vec3i s=map->floatToInt(start),g=map->floatToInt(goal);
    DMP::GraphSearch cpu(field.potential.data(),dim.x(),dim.y(),dim.z(),weight,cweight);
    DMP::GraphSearch cached(field.potential.data(),dim.x(),dim.y(),dim.z(),weight,cweight,false,field.legal_moves.data());
    const double a=cpu.plan(s.x(),s.y(),s.z(),g.x(),g.y(),g.z());
    const double b=cached.plan(s.x(),s.y(),s.z(),g.x(),g.y(),g.z()); compareGraphs(cpu,a,cached,b);
    std::ofstream output(mask_path,std::ios::binary);
    output.write(reinterpret_cast<const char*>(field.legal_moves.data()),field.legal_moves.size()*sizeof(uint32_t));
    require(bool(output),"frozen mask write failed");
    std::cout<<std::setprecision(17)<<"{\"all_states_exact\":true,\"cost\":"<<a
             <<",\"raw_points\":"<<cpu.getPath().size()<<",\"closed\":"<<cpu.getCloseSet().size()
             <<",\"discovered\":"<<cpu.getAllSet().size()<<",\"mask_cells\":"<<field.legal_moves.size()<<"}\n";
}
#endif

int main(int argc,char** argv) {
    if(argc<2)return 2;
    try {
        const std::string name=argv[1];
        if(name=="cpu_equivalence")cpuEquivalence();
        else if(name=="lifecycle")lifecycle();
#ifdef PATHCOVER_TEST_CUDA
        else {
            int devices=0; const cudaError_t status=cudaGetDeviceCount(&devices);
            if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || (status==cudaSuccess && devices==0))return 77;
            require(status==cudaSuccess,"CUDA device query failed");
            if(name=="gpu_masks")gpuMasks();
            else if(name=="gpu_equivalence")gpuEquivalence();
            else if(name=="frozen" && argc==5){frozen(argv[2],argv[3],argv[4]);return 0;}
            else return 3;
        }
#else
        else return 3;
#endif
        std::cout<<name<<" passed\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
