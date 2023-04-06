#ifndef STELLA_VSLAM_DATA_BOW_DATABASE_H
#define STELLA_VSLAM_DATA_BOW_DATABASE_H

#include "stella_vslam/data/bow_vocabulary.h"
#include "stella_vslam/data/base_place_recognition.h"

#include <mutex>
#include <list>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace stella_vslam::data {

class frame;
class keyframe;

class bow_database : public base_place_recognition {
public:
    /**
     * Constructor
     * @param bow_vocab
     */
    explicit bow_database(bow_vocabulary* bow_vocab);

    /**
     * Destructor
     */
    ~bow_database() override;

    /**
     * Add a keyframe to the database
     * @param keyfrm
     */
    void add_keyframe(const std::shared_ptr<keyframe>& keyfrm) override;

    /**
     * Erase the keyframe from the database
     * @param keyfrm
     */
    void erase_keyframe(const std::shared_ptr<keyframe>& keyfrm) override;

    /**
     * Clear the database
     */
    void clear() override;

    bow_vocabulary* getBowVocab() const;

    /**
     *
     * @param keyfrm [in,out]
     */
    void computeRepresentation(const std::shared_ptr<keyframe>& keyfrm) override;

    /**
     * Compute representation BoW or global desc
     * @param frame
     */
    void computeRepresentation(data::frame& frame, const cv::Mat& img) override;

    /**
     * Acquire keyframes over score
     */
    std::vector<std::shared_ptr<keyframe>> acquire_keyframes(const bow_vector& bow_vec, const float min_score = 0.0f,
                                                             const std::set<std::shared_ptr<keyframe>>& keyfrms_to_reject = {});

protected:

    /**
     * Compute the number of shared words
     * @param bow_vec
     * @param keyfrms_to_reject
     * @return number of shared words between the query and the each of keyframes contained in the database (key: keyframes that share word with query keyframe, value: number of shared words)
     */
    std::unordered_map<std::shared_ptr<keyframe>, unsigned int>
    compute_num_common_words(const bow_vector& bow_vec,
                             const std::set<std::shared_ptr<keyframe>>& keyfrms_to_reject = {}) const;

    /**
     * Compute scores (scores_) between the query and the each of keyframes contained in the database
     * @param num_common_words
     * @param bow_vec
     * @param min_num_common_words_thr
     * @return similarity scores between the query and the each of keyframes contained in the database (key: keyframes that share word with query keyframe, value: score)
     */
    std::unordered_map<std::shared_ptr<keyframe>, float>
    compute_scores(const std::unordered_map<std::shared_ptr<keyframe>, unsigned int>& num_common_words,
                   const bow_vector& bow_vec,
                   const unsigned int min_num_common_words_thr,
                   const float min_score,
                   float& best_score) const;

    //! BoW database
    std::unordered_map<unsigned int, std::list<std::shared_ptr<keyframe>>> keyfrms_in_node_{};

    //-----------------------------------------
    // BoW vocabulary
    //! BoW vocabulary
    bow_vocabulary* bow_vocab_;
};

} // namespace stella_vslam

#endif // STELLA_VSLAM_DATA_BOW_DATABASE_H
