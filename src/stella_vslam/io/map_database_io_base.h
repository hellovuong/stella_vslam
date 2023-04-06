#ifndef STELLA_VSLAM_IO_MAP_DATABASE_IO_BASE_H
#define STELLA_VSLAM_IO_MAP_DATABASE_IO_BASE_H

#include "stella_vslam/data/bow_vocabulary.h"

#include <string>

namespace stella_vslam {

namespace data {
class camera_database;
class bow_database;
class map_database;
} // namespace data

namespace io {

class map_database_io_base {
public:
    /**
     * Save the map database
     */
    virtual bool save(const std::string& path,
                      const data::camera_database* cam_db,
                      const data::orb_params_database* orb_params_db,
                      const data::map_database* map_db)
        = 0;

    /**
     * Load the map database
     */
    virtual bool load(const std::string& path,
                      data::camera_database* cam_db,
                      data::orb_params_database* orb_params_db,
                      data::map_database* map_db,
                      data::base_place_recognition* vpr_db)
        = 0;
};

} // namespace io
} // namespace stella_vslam

#endif // STELLA_VSLAM_IO_MAP_DATABASE_IO_BASE_H
