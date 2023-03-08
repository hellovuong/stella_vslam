//
// Created by vuong on 2/28/23.
//

#ifndef STELLA_VSLAM_HLOC_DATABASE_H
#define STELLA_VSLAM_HLOC_DATABASE_H

#include <opencv2/core.hpp>
#include <map>

#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/hloc/hloc.h"

namespace stella_vslam {

/// Id of entries of the database
typedef int EntryId;

class Result {
public:
    EntryId Id{};
    double Score{};

    /**
     * Empty constructors
     */
    inline Result() = default;

    /**
     * Creates a result with the given data
     * @param _id entry id
     * @param _score score
     */
    inline Result(EntryId _id, double _score)
        : Id(_id), Score(_score) {}

    /**
     * Compares the scores of two results
     * @return true iff this.score < r.score
     */
    inline bool operator<(const Result& r) const {
        return this->Score < r.Score;
    }

    /**
     * Compares the scores of two results
     * @return true iff this.score > r.score
     */
    inline bool operator>(const Result& r) const {
        return this->Score > r.Score;
    }

    /**
     * Compares the entry id of the result
     * @return true iff this.id == id
     */
    inline bool operator==(EntryId id) const {
        return this->Id == id;
    }

    /**
     * Compares the score of this entry with a given one
     * @param s score to compare with
     * @return true iff this score < s
     */
    inline bool operator<(double s) const {
        return this->Score < s;
    }

    /**
     * Compares the score of this entry with a given one
     * @param s score to compare with
     * @return true iff this score > s
     */
    inline bool operator>(double s) const {
        return this->Score > s;
    }

    /**
     * Compares the score of two results
     * @param a
     * @param b
     * @return true iff a.Score > b.Score
     */
    static inline bool gt(const Result& a, const Result& b) {
        return a.Score > b.Score;
    }

    /**
     * Compares the scores of two results
     * @return true iff a.Score > b.Score
     */
    inline static bool ge(const Result& a, const Result& b) {
        return a.Score > b.Score;
    }

    /**
     * Returns true iff a.Score >= b.Score
     * @param a
     * @param b
     * @return true iff a.Score >= b.Score
     */
    static inline bool geq(const Result& a, const Result& b) {
        return a.Score >= b.Score;
    }

    /**
     * Returns true iff a.Score >= s
     * @param a
     * @param s
     * @return true iff a.Score >= s
     */
    static inline bool geqv(const Result& a, double s) {
        return a.Score >= s;
    }

    /**
     * Returns true iff a.Id < b.Id
     * @param a
     * @param b
     * @return true iff a.Id < b.Id
     */
    static inline bool ltId(const Result& a, const Result& b) {
        return a.Id < b.Id;
    }
};

namespace hloc {
class keyframe {
public:
    explicit keyframe(std::shared_ptr<data::keyframe> cur_keyfrm);
    //! current keyframe
    std::shared_ptr<data::keyframe> keyfrm_{};
    void computeNew(std::vector<cv::Point2f>& keypoint, std::vector<float>& scores);
    cv::Mat global_descriptors_{};
};
} // namespace hloc

typedef std::list<std::pair<std::shared_ptr<hloc::keyframe>, double>> query_res_t;

class hloc_database {
public:
    void add(const std::shared_ptr<hloc::keyframe>& keyfrm, cv::Mat& new_global_descriptor) {
        database[keyfrm] = new_global_descriptor;
    }

    query_res_t query(cv::Mat& global_descriptor) {
        query_res_t ret;
        for (auto const& desc : database) {
            auto key_frame = desc.first;
            if ((not key_frame)
                or (not key_frame->keyfrm_)
                or (key_frame->keyfrm_->will_be_erased())) {
                continue;
            }
            auto score = global_descriptor.dot(desc.second);
            if (score < 0.7)
                continue ;
            ret.push_back(std::make_pair(key_frame, score));
        }
        ret.sort(comp_first);
        return ret;
    }

private:
    std::unordered_map<std::shared_ptr<hloc::keyframe>, cv::Mat> database{};
    static bool comp_first(const std::pair<std::shared_ptr<hloc::keyframe>, double>& a,
                           const std::pair<std::shared_ptr<hloc::keyframe>, double>& b) {
        return a.first > b.first;
    }
};
} // namespace stella_vslam
#endif // STELLA_VSLAM_HLOC_DATABASE_H
