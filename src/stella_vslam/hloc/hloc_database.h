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

using namespace std;

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

class hloc_database {
public:
    void add(cv::Mat& new_global_descriptor) {
        database[database.size()] = new_global_descriptor;
    }

    void query(cv::Mat& global_descriptor, vector<Result>& ret) {
        query(global_descriptor, ret, 0, database.size());
    }

    void query(cv::Mat& global_descriptor, vector<Result>& ret, EntryId start_id, EntryId end_id) {
        for (int i = start_id; i < end_id; i++) {
            ret.emplace_back(i, global_descriptor.dot(database[i]));
        }
        sort(ret.begin(), ret.end(), greater<>());
    }

private:
    map<EntryId, cv::Mat> database;
};

namespace hloc {
class keyframe {
public:
    explicit keyframe(std::shared_ptr<data::keyframe> cur_keyfrm);
    //! current keyframe
    std::shared_ptr<data::keyframe> keyfrm_{};

    void computeWindow();
    void computeNew(vector<cv::Point2f>& keypoint, std::vector<float>& scores);

    vector<cv::Point3f> point_3d{};
    vector<cv::Point2f> point_2d_uv_{};
    vector<cv::Point2f> point_2d_norm{};
    vector<double> point_id{};
    vector<cv::KeyPoint> keypoints_{};
    vector<cv::KeyPoint> keypoints_norm_{};
    vector<cv::KeyPoint> window_keypoints{};
    std::vector<float> scores_{};
    std::vector<float> window_scores{};
    cv::Mat local_descriptors_{};
    cv::Mat window_local_descriptors{};
    cv::Mat global_descriptors_{};
};
} // namespace hloc

} // namespace stella_vslam
#endif // STELLA_VSLAM_HLOC_DATABASE_H
