// Same occupied cells as IPC's offset loops, with contiguous z writes and NumPy input.
void LoaclAstarClass::setObsMatrixFast(
    const Eigen::Ref<const Eigen::Matrix<double,Eigen::Dynamic,3,Eigen::RowMajor>>& cloud, double radius) {
    if(!std::isfinite(radius)||radius<0) throw std::invalid_argument("Invalid inflation radius");
    const size_t cells=size_t(GL_SIZE_(0))*GL_SIZE_(1)*GL_SIZE_(2);
    packed_.clear();
    memset(grid_map_,0,cells);
    std::vector<double> offsets;
    for(double d=-radius;d<=radius;d+=resolution_) offsets.push_back(d);
    std::vector<int> indices[3];
    for(auto &v:indices) v.reserve(offsets.size());
    for(Eigen::Index row=0;row<cloud.rows();++row){
        Eigen::Vector3d point=cloud.row(row).transpose();
        if(!point.allFinite()) throw std::invalid_argument("Nonfinite map point");
        if(!coorIsInMap(point)) continue;
        for(int axis=0;axis<3;++axis){
            auto &v=indices[axis];v.clear();
            for(double d:offsets){
                const double q=point[axis]+d-center_[axis];
                if(q<gl_low_[axis] || q>=gl_upp_[axis]) continue;
                const int index=static_cast<int>((q-gl_low_[axis])/resolution_);
                if(v.empty() || v.back()!=index) v.push_back(index);
            }
        }
        const auto &z=indices[2];
        for(int x:indices[0]) for(int y:indices[1]){
            for(size_t j=0;j<z.size();){
                const int begin=z[j];int end=begin;++j;
                while(j<z.size() && z[j]<=end+1) end=z[j++];
                memset(grid_map_+(size_t(x)*GL_SIZE_(1)+y)*GL_SIZE_(2)+begin,1,end-begin+1);
            }
        }
    }
}
